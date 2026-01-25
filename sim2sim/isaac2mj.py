# import sys
# import os
# import torch
# import mujoco
# import mujoco.viewer
# import time
# import argparse
# import pickle
# import numpy as np
# # import obs_save
# # 加载 mujoco 模型
# m = mujoco.MjModel.from_xml_path('sim2sim/scence.xml')
# d = mujoco.MjData(m)

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
# from my_utils import gamepad

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #储存观测状态
# # obs_saver = obs_save.TensorTypeSaver(save_dir="./obs_data", device="cpu")

# def get_sensor_data(sensor_name):
#     sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
#     if sensor_id == -1:
#         raise ValueError(f"Sensor '{sensor_name}' not found in model!")
#     start_idx = m.sensor_adr[sensor_id]
#     dim = m.sensor_dim[sensor_id]
#     sensor_values = d.sensordata[start_idx : start_idx + dim]
#     return torch.tensor(
#         sensor_values, 
#         device=device, 
#         dtype=torch.float32
#     )

# def set_joint_angle(joint_name, angle):
#     joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
#     d.qpos[m.jnt_qposadr[joint_id]] = angle
    
# def world2self(quat, v):
#     q_w = quat[0] 
#     q_vec = quat[1:] 
#     v_vec = torch.tensor(v, device=device,dtype=torch.float32)
#     a = v_vec * (2.0 * q_w**2 - 1.0)
#     b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
#     c = q_vec * torch.dot(q_vec, v_vec) * 2.0
#     result = a - b + c
#     return result.to(device)

# def O_get_obs(env_cfg, obs_scales, actions, default_dof_pos, commands=[0.0, 0.0, 0.0, 0.22]):
#     commands_scale = torch.tensor(
#         [obs_scales["lin_vel"], obs_scales["lin_vel"], 
#          obs_scales["ang_vel"], obs_scales["height_measurements"]], device=device, dtype=torch.float32)
#     base_quat = get_sensor_data("orientation")
#     gravity = [0.0, 0.0, -1.0]
#     projected_gravity = world2self(base_quat,torch.tensor(gravity, device=device, dtype=torch.float32))
#     base_lin_vel = world2self(base_quat,get_sensor_data("base_lin_vel"))
#     base_ang_vel = get_sensor_data("base_ang_vel")
#     dof_pos = torch.zeros(env_cfg["num_actions"], device=device, dtype=torch.float32)
#     for i, dof_name in enumerate(env_cfg["dof_names"]):
#         dof_pos[i] = get_sensor_data(dof_name+"_p")[0]
#         if i==3:
#             break
#     dof_vel = torch.zeros(env_cfg["num_actions"], device=device, dtype=torch.float32)
#     for i, dof_name in enumerate(env_cfg["dof_names"]):
#         dof_vel[i] = get_sensor_data(dof_name+"_v")[0]

#     cmds = torch.tensor(commands, device=device, dtype=torch.float32)

#     print("base_lin_vel:", base_lin_vel)
#     print("base_ang_vel:", base_ang_vel)
#     print("projected_gravity:", projected_gravity)
#     print("dof_pos:", dof_pos)
#     print("dof_vel:", dof_vel)
#     print("commands:", commands)
#     # obs_saver.add_tensor("base_lin_vel", base_lin_vel)
#     # obs_saver.add_tensor("base_ang_vel", base_ang_vel)
#     # obs_saver.add_tensor("projected_gravity", projected_gravity)
#     # obs_saver.add_tensor("dof_vel", dof_vel)
#     # obs_saver.add_tensor("dof_pos", dof_pos[0:4])
#     return torch.cat(
#         [
#             # base_lin_vel * obs_scales["lin_vel"],  # 3
#             base_ang_vel * obs_scales["ang_vel"],  # 3
#             projected_gravity,  # 3
#             cmds * commands_scale,  # 4
#             (dof_pos[0:4] - default_dof_pos[0:4]) * obs_scales["dof_pos"],  # 4
#             dof_vel * obs_scales["dof_vel"],  # 6
#             actions,  # 6
#         ],
#         axis=-1,
#     )

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-e", "--exp_name", type=str, default="Dog-v25.11.19")
#     args = parser.parse_args()

#     # 拼接到 logs 文件夹的路径
#     log_dir = os.path.join('./logs', args.exp_name)
#     cfg_path = os.path.join(log_dir, 'cfgs.pkl')

#     # 读取配置文件
#     if os.path.exists(cfg_path):
#         print("文件存在:", cfg_path)
#         env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
#     else:
#         print("文件不存在:", cfg_path)
#         exit()

#     # 加载游戏控制器
#     pad = gamepad.control_gamepad(command_cfg)
#     commands, reset_flag = pad.get_commands()

#     # 加载模型
#     try:
#         loaded_policy = torch.jit.load(os.path.join(log_dir, "policy.pt"))
#         # loaded_policy = torch.jit.load("policy.pt")
#         loaded_policy.eval()  # 设置为评估模式
#         loaded_policy.to(device)
#         print("模型加载成功!")
#     except Exception as e:
#         print(f"模型加载失败: {e}")
#         exit()

#     #dof limits
#     lower = [env_cfg["dof_limit"][name][0] for name in env_cfg["dof_names"]]
#     upper = [env_cfg["dof_limit"][name][1] for name in env_cfg["dof_names"]]
#     dof_pos_lower = torch.tensor(lower).to(device)
#     dof_pos_upper = torch.tensor(upper).to(device)
        
#     # 初始化观察数据
#     history_obs_buf = torch.zeros((obs_cfg["history_length"], obs_cfg["num_slice_obs"]), device=device, dtype=torch.float32)
#     slice_obs_buf = torch.zeros(obs_cfg["num_slice_obs"], device=device, dtype=torch.float32)
#     obs_buf = torch.zeros((obs_cfg["num_obs"]), device=device, dtype=torch.float32)
#     default_dof_pos = torch.tensor(
#         [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
#         device=device,
#         dtype=torch.float32)
#     actions = torch.zeros((env_cfg["num_actions"]), device=device, dtype=torch.float32)

#     # 启动 mujoco 渲染
#     with mujoco.viewer.launch_passive(m, d) as viewer:
#         while viewer.is_running():
#             # slice_obs_buf = get_obs(env_cfg=env_cfg, obs_scales=obs_cfg["obs_scales"],
#             #                         actions=actions, default_dof_pos=default_dof_pos, commands=commands)
#             # slice_obs_buf = slice_obs_buf.unsqueeze(0)
#             # obs_buf = torch.cat([history_obs_buf, slice_obs_buf], dim=0).view(-1)
#             # # 更新历史缓冲区
#             # if obs_cfg["history_length"] > 1:
#             #     history_obs_buf[:-1, :] = history_obs_buf[1:, :].clone()  # 移位操作
#             # history_obs_buf[-1, :] = slice_obs_buf 
#             # actions = loaded_policy(obs_buf)
#             # actions = torch.clip(actions, -env_cfg["clip_actions"], env_cfg["clip_actions"])
#             # # 更新动作
#             # target_dof_pos = actions[0:4] * env_cfg["joint_action_scale"] + default_dof_pos[0:4]
#             # target_dof_vel = actions[4:6] * env_cfg["wheel_action_scale"]
#             # target_dof_pos = torch.clamp(target_dof_pos, dof_pos_lower[0:4],dof_pos_upper[0:4])
#             # # print("act:", act)
#             # for i in range(env_cfg["num_actions"]-2):
#             #     d.ctrl[i] = target_dof_pos.detach().cpu().numpy()[i]

#             # d.ctrl[4] = target_dof_vel.detach().cpu().numpy()[0]
#             # d.ctrl[5] = target_dof_vel.detach().cpu().numpy()[1]
            

#             # 获取控制命令
#             commands, reset_flag = pad.get_commands()
#             if reset_flag:
#                 mujoco.mj_resetData(m, d)

#             # 执行一步模拟
#             step_start = time.time()
#             for i in range(5):
#                 mujoco.mj_step(m, d)
#             # 更新渲染
#             viewer.sync()
#             # 同步时间
#             time_until_next_step = m.opt.timestep*5 - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)
#     print("close viewer")
#     # obs_saver.save_all()

# if __name__ == "__main__":
#     main()

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------

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

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    loaded_policy: torch.jit.RecursiveScriptModule = torch.jit.load(os.path.join(log_dir, "exported", "policy.pt"))
    loaded_policy.eval()  # 设置为评估模式
    loaded_policy.to(device)
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")


# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path('sim2sim/scence.xml')
d = mujoco.MjData(m)


def world2self(quat, v):
    q_w = quat[0] 
    q_vec = quat[1:] 
    v_vec = torch.tensor(v, device=device,dtype=torch.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result.to(device)

def get_sensor_data(sensor_name: str):
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
                # self.robot.data.root_ang_vel_b,
                # self.robot.data.projected_gravity_b,   
                # feet_contact_forces,
                # self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                # self.robot.data.joint_vel,
                # self._actions,
                # self.commands,

sensor = ["orientation"]
obs_sensor = ["ang_vel", "foots_contact", "joint_pos", "joint_vel", "projected_gravity",]
calf_joints_contact_sensor = ["FL_calf_joint_contact", "FR_calf_joint_contact", "RL_calf_joint_contact", "RR_calf_joint_contact"]

def get_obs(last_actins: torch.Tensor = None, commands: torch.Tensor = None):
    dof_pos = torch.zeros(env_cfg["action_space"], device=device, dtype=torch.float32)
    for i, dof_name in enumerate(env_cfg["joints_names"]):
        dof_pos[i] = get_sensor_data(dof_name+"_p")

    dof_vel = torch.zeros(env_cfg["action_space"], device=device, dtype=torch.float32)
    for i, dof_name in enumerate(env_cfg["joints_names"]):
        dof_vel[i] = get_sensor_data(dof_name+"_v")

    # foots_contact = torch.zeros(4, device=device, dtype=torch.float32)
    # for i, sensor_name in enumerate(calf_joints_contact_sensor):
    #     foots_contact[i] = get_sensor_data(sensor_name)

    quat = get_sensor_data("orientation")
    projected_gravity = world2self(quat, [0, 0, -1])
    projected_gravity = projected_gravity/torch.linalg.norm(projected_gravity)
    ang_vel = get_sensor_data("base_ang_vel")
    # print("ang_vel:",projected_gravity)

    result = torch.cat((ang_vel, dof_pos, dof_vel, projected_gravity, last_actins, commands), dim=0).unsqueeze(0)
    # print("result:",result)
    return result


class Env(VecEnv):
    def __init__(self, env_cfg, num_envs = 1):
        self.num_envs = num_envs
        self.num_actions = env_cfg["action_space"]

    def get_observations(self) -> TensorDict:
        return TensorDict({
            "privileged": torch.zeros(1,env_cfg["state_space"]-env_cfg["observation_space"], dtype=torch.int8, device=device),
            "policy": torch.zeros((1,env_cfg["observation_space"],), dtype=torch.int8, device=device),
            })

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        pass




def main():
    # 加载控制器
    pad = gamepad.control_gamepad(env_cfg["command_cfg"])
    # 初始化观测
    obs = torch.zeros((1,env_cfg["observation_space"]), dtype=torch.float32, device=device)

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
    torch.jit.RecursiveScriptModule

    # print(loaded_policy)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    print(f"batch_first: {loaded_policy.rnn.batch_first}")
    print(f"input_size: {loaded_policy.rnn.input_size}")
    
    
    
    # 启动mujoco渲染
    decimation: int = 5
    hidden = None
    with mujoco.viewer.launch_passive(m, d) as viewer:
        with torch.no_grad():
            while viewer.is_running():
                actions = loaded_policy(obs)
                ex_actions = torch.clip(actions, min=-env_cfg["clip_actions"], max=env_cfg["clip_actions"])
                target_dof_pos = ex_actions * env_cfg["action_scale"] + default_dof_pos
                target_dof_pos = torch.clamp(target_dof_pos, dof_pos_lower, dof_pos_upper)
                to_cpu = target_dof_pos.detach().squeeze().cpu().numpy()
                for i in range(env_cfg["action_space"]):
                    d.ctrl[i] = to_cpu[i]

                # # 获取命令
                commands, reset_flag = pad.get_commands()
                if reset_flag:
                    mujoco.mj_resetData(m, d)

                # # 获取观测
                obs = get_obs(
                    ex_actions.squeeze(),
                    torch.tensor(commands, dtype=torch.float32, device=device),
                )
                # time.sleep(2)

                # 执行模拟
                step_start = time.time()
                for i in range(decimation):
                    mujoco.mj_step(m, d)
                # 更新渲染
                viewer.sync()
                # 同步时间
                time_until_next_step = m.opt.timestep * decimation - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    print("close viewer")

if __name__ == "__main__":
    main()
