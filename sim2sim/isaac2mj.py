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

# import rsl_rl
# from rsl_rl.algorithms import PPO
# from rsl_rl.env import VecEnv
# from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
# from rsl_rl.utils import resolve_obs_groups, store_code_state
# from rsl_rl.runners import DistillationRunner, OnPolicyRunner


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

        # "lin_vel": 2.0,
        # "ang_vel": 1.0,
        # "dof_pos": 1.0,
        # "dof_vel": 0.1,
        # "lin_acc": 0.1,

class GetObservation: 
    def __init__(self, env_cfg, device):
        self.projected_gravity = torch.zeros((1, 4), device=device, dtype=torch.float32)
        self.foots_contact = torch.zeros((1, 4), device=device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((1, 3), device=device, dtype=torch.float32)
        self.slice_obs   = torch.zeros((1, env_cfg["env_cfg"]["policy_slice_obs"]), device=device, dtype=torch.float32)
        self.history_obs = torch.zeros((1, env_cfg["env_cfg"]["history_length"], env_cfg["env_cfg"]["policy_slice_obs"]), device=device, dtype=torch.float32)
        self.obs_buf = torch.zeros((1, env_cfg["env_cfg"]["observation_space"]), device=device, dtype=torch.float32)
        self.dof_pos = torch.zeros((1, env_cfg["action_space"]), device=device, dtype=torch.float32)
        self.dof_vel = torch.zeros((1, env_cfg["action_space"]), device=device, dtype=torch.float32)
        self.lin_vel = torch.zeros((1, 3), device=device, dtype=torch.float32)
        self.base_acc = torch.zeros((1, 3), device=device, dtype=torch.float32)

        self.obs_scales = env_cfg["obs_scales"]

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
            self.dof_pos[:,i] = self.get_sensor_data(dof_name+"_p")[0] * self.obs_scales["dof_pos"]

        for i, dof_name in enumerate(env_cfg["joints_names"]):
            self.dof_vel[:,i] = self.get_sensor_data(dof_name+"_v")[0] * self.obs_scales["dof_vel"]

        quat = self.get_sensor_data("orientation")
        self.projected_gravity = self.world2self(quat, [0.0, 0.0, -1.0]).unsqueeze(0)
        self.base_ang_vel = self.get_sensor_data("base_ang_vel").unsqueeze(0) * self.obs_scales["ang_vel"]
        self.lin_vel = self.get_sensor_data("base_lin_vel").unsqueeze(0) * self.obs_scales["lin_vel"]
        # self.base_acc = self.get_sensor_data("base_acc").unsqueeze(0)
        
        self.slice_obs = torch.cat((self.lin_vel, self.base_ang_vel, self.projected_gravity, self.dof_pos, self.dof_vel), dim=1)
        self.obs_buf = torch.cat([self.history_obs, self.slice_obs.unsqueeze(1)], dim=1).view(1, -1)
        self.obs_buf = torch.cat([self.obs_buf, last_actins], dim=-1)
        self.obs_buf = torch.cat([self.obs_buf, torch.tensor(commands, dtype=torch.float32, device=device).unsqueeze(0)], dim=-1)

        clip_obs = self.obs_scales["clip_observations"]
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if env_cfg["env_cfg"]["history_length"] > 1:
            self.history_obs[:, :-1, :] = self.history_obs[:, 1:, :].clone()
        self.history_obs[:, -1, :] = self.slice_obs.clone()

        return self.obs_buf

def main(log_dir):
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
        loaded_policy = torch.jit.load(f"{log_dir}/exported/policy.pt")
        loaded_policy.eval()
        loaded_policy.to('cuda')
        print("模型加载成功!")
        print(loaded_policy)
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()


    # 启动mujoco渲染
    decimation: int = 10
    last_processed_actions = torch.zeros((1, 12), device=device, dtype=torch.float32)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 加载控制器
        pad = gamepad.control_gamepad(env_cfg["command_cfg"])
        with torch.no_grad():
            obs = _obs.get_obs(torch.zeros((1, env_cfg["action_space"]), device=device, dtype=torch.float32), pad.get_commands()[0])
            while viewer.is_running():
                actions = loaded_policy(obs)
                ex_actions = torch.clip(actions, min=-env_cfg["clip_actions"], max=env_cfg["clip_actions"])
                target_dof_pos = ((ex_actions * env_cfg["action_scale"]) + default_dof_pos - (_obs.dof_pos / _obs.obs_scales["dof_pos"]))
                # target_dof_pos = torch.clip(target_dof_pos, dof_pos_lower, dof_pos_upper
                # target_dof_pos = torch.clamp(target_dof_pos, dof_pos_lower, dof_pos_upper)
                # processed_actions = target_dof_pos*0.7 + last_processed_actions*0.3
                # last_processed_actions = processed_actions.clone()
                to_cpu = target_dof_pos.cpu().numpy()

                for i in range(env_cfg["action_space"]):
                    d.ctrl[i] = to_cpu[0, i]
                print(d.ctrl.shape)
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
                commands, reset_flag, _, _ = pad.get_commands()
                # print(commands)
                if reset_flag:
                    mujoco.mj_resetData(m, d)

                obs = _obs.get_obs(ex_actions, commands)
                # obs = obs + torch.normal(mean=0.0, std=0.008, size=(1, env_cfg["env_cfg"]["observation_space"]), device=device)

    print("close viewer")

if __name__ == "__main__":
    main(log_dir)
