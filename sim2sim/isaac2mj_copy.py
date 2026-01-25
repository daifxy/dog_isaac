import sys
import os
import torch
from tensordict import TensorDict
import datetime
import time
import sys
import yaml
import argparse
import re

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.runners import DistillationRunner, OnPolicyRunner


import mujoco
import mujoco.viewer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from my_utils import gamepad

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("-e", "--exp_name", type=str, default="Dog", help="Name of the experiment.")
parser.add_argument("-r", "--run_name", type=str, default=None, help="Name of the run.")
parser.add_argument("-p", "--abs_path", type=str, default=None, help="Path to the run_file, equal to .../exp_name + run_name.")
args = parser.parse_args()

# 加载配置文件
if args.abs_path is None:
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "rsl_rl", args.exp_name, args.run_name)
else:
    log_dir = args.abs_path

if os.path.exists(log_dir):
    with open(os.path.join(log_dir, "params", "env.yaml"), "r") as f:
        env_cfg = yaml.load(stream=f, Loader=yaml.UnsafeLoader)
    with open(os.path.join(log_dir, "params", "agent.yaml"), "r") as f:
        agent_cfg = yaml.load(stream=f, Loader=yaml.UnsafeLoader)
else:
    print(f"文件不存在:{log_dir}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path('sim2sim/scence.xml')
d = mujoco.MjData(m)

class GetObservation: 
    def __init__(self, env_cfg, device):
        self.projected_gravity = torch.zeros((1, 4), device=device, dtype=torch.float32)
        self.foots_contact = torch.zeros((1, 4), device=device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((1, 3), device=device, dtype=torch.float32)
        self.slice_obs   = torch.zeros((1, env_cfg["env_cfg"]["policy_obs"]), device=device, dtype=torch.float32)
        self.history_obs = torch.zeros((1, env_cfg["env_cfg"]["history_length"], env_cfg["env_cfg"]["policy_obs"]), device=device, dtype=torch.float32)
        self.obs_buf = torch.zeros((1, env_cfg["env_cfg"]["observation_space"]), device=device, dtype=torch.float32)
        self.dof_pos = torch.zeros((1, env_cfg["action_space"]), device=device, dtype=torch.float32)
        self.dof_vel = torch.zeros((1, env_cfg["action_space"]), device=device, dtype=torch.float32)

    def world2self(self, quat, v)-> torch.Tensor:
        q_w = quat[0] 
        q_vec = quat[1:] 
        v_vec = torch.tensor(v, device=device,dtype=torch.float32)
        a = v_vec * (2.0 * q_w**2 - 1.0)
        b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
        c = q_vec * torch.dot(q_vec, v_vec) * 2.0
        result = a - b + c
        return result.to(device)

    def get_sensor_data(self, sensor_name: str):
        sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = m.sensor_adr[sensor_id]
        dim = m.sensor_dim[sensor_id]
        sensor_values = d.sensordata[start_idx : start_idx + dim]
        return torch.tensor(
            sensor_values, 
            device=device, 
            dtype=torch.float32
        )

    def get_obs(self, last_actins: torch.Tensor = None, commands = None):
        for i, dof_name in enumerate(env_cfg["joints_names"]):
            self.dof_pos[:,i] = self.get_sensor_data(dof_name+"_p")[0]

        for i, dof_name in enumerate(env_cfg["joints_names"]):
            self.dof_vel[:,i] = self.get_sensor_data(dof_name+"_v")[0]

        quat = self.get_sensor_data("orientation")
        self.projected_gravity = self.world2self(quat, [0.0, 0.0, -1.0]).unsqueeze(0)
        self.base_ang_vel = self.get_sensor_data("base_ang_vel").unsqueeze(0)

        # print(self.projected_gravity)
        # print(torch.norm(self.projected_gravity,dim=-1,keepdim=True))
        # print(self.base_ang_vel)
        # print(self.dof_pos,"XXXXXXXXX")
        if (self.dof_vel >=6.5).any():
            print(self.dof_vel[(self.dof_vel >=6.5)],"YYYYYYYYY")
        # print(last_actins.shape)
        self.slice_obs = torch.cat((self.base_ang_vel, self.projected_gravity, self.dof_pos, self.dof_vel, last_actins), dim=1)

        self.obs_buf = torch.cat([self.history_obs, self.slice_obs.unsqueeze(1)], dim=1).view(1, -1)
        obs_buf = torch.cat([self.obs_buf, torch.tensor(commands, dtype=torch.float32, device=device).unsqueeze(0)], dim=-1)

        if env_cfg["env_cfg"]["history_length"] > 1:
            self.history_obs[:, :-1, :] = self.history_obs[:, 1:, :].clone()
            self.history_obs[:, -1, :] = self.slice_obs.clone()

        return obs_buf

def main():
    # 加载控制器
    pad = gamepad.control_gamepad(env_cfg["command_cfg"])
    # 初始化观测
    _obs = GetObservation(env_cfg, device)

    #dof limits
    lower = [env_cfg["joint_pos_limits"][name][0] for name in env_cfg["joints_names"]]
    upper = [env_cfg["joint_pos_limits"][name][1] for name in env_cfg["joints_names"]]
    dof_pos_lower = torch.tensor(lower).to(device)
    dof_pos_upper = torch.tensor(upper).to(device)

    # default_dof_pos
    default_dof_pos = []
    for name in env_cfg["joints_names"]:
        for keys, value in env_cfg["robot_cfg"]["init_state"]["joint_pos"].items():
            if re.search(keys, name):
                default_dof_pos.append(value)
    default_dof_pos = torch.tensor(default_dof_pos, device=device, dtype=torch.float32)

    try:
        loaded_policy = torch.jit.load("/home/du/dog_baseon_isaac/logs/rsl_rl/Dog-test/26.01.02-19.45/exported/policy.pt")
        loaded_policy.eval()
        loaded_policy.to('cuda')
        print("模型加载成功!")
        print(loaded_policy)
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()


    # 启动mujoco渲染
    decimation: int = 10
    previous_actions = torch.zeros((3, env_cfg["action_space"]), device=device, dtype=torch.float32)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        with torch.no_grad():
            obs = _obs.get_obs(torch.zeros((1, env_cfg["action_space"]), device=device, dtype=torch.float32), pad.get_commands()[0])
            while viewer.is_running():
                actions = loaded_policy(obs)
                actions = actions + torch.normal(mean=0.0, std=0.008, size=(1, env_cfg["action_space"]), device=device)
                ex_actions = torch.clip(actions, min=-env_cfg["clip_actions"], max=env_cfg["clip_actions"])
                target_dof_pos = (ex_actions * env_cfg["action_scale"]) + default_dof_pos
                # target_dof_pos = torch.clamp(target_dof_pos, dof_pos_lower, dof_pos_upper)
                previous_actions[:-1] = previous_actions[1:].clone()
                previous_actions[0] = target_dof_pos
                to_cpu = previous_actions.detach().cpu().numpy()
                # print(to_cpu.shape)

                # action = [0.0, -0.38, 1.3, 0.0, -0.38, 1.3, 0.0, -0.33, 1.3, 0.0, -0.33, 1.3]
                random_num = torch.rand(1)
                # if random_num < 0.8:
                for i in range(env_cfg["action_space"]):
                    d.ctrl[i] = to_cpu[0, i]
                # elif random_num >= 0.8 and random_num < 0.9:
                    # for i in range(env_cfg["action_space"]):
                        # d.ctrl[i] = to_cpu[1, i]
                # elif random_num >= 0.9:
                    # for i in range(env_cfg["action_space"]):
                        # d.ctrl[i] = to_cpu[2, i]
                
                step_start = time.time()
                # 执行模拟
                for i in range(decimation):
                    mujoco.mj_step(m, d)
                    # 更新渲染
                viewer.sync()


                # 同步时间
                time_until_next_step = m.opt.timestep * decimation - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
                # 获取新的观测
                commands, reset_flag = pad.get_commands()
                # print(commands)
                if reset_flag:
                    mujoco.mj_resetData(m, d)

                obs = _obs.get_obs(ex_actions, commands)
                obs = obs + torch.normal(mean=0.0, std=0.008, size=(1, env_cfg["env_cfg"]["observation_space"]), device=device)

    print("close viewer")

if __name__ == "__main__":
    main()
