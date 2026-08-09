# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs import mdp  # noqa: F401, F403

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg


def force_body_to_world_torch(force_body, q_body_to_world):
    """
    PyTorch 批量版本，支持 GPU 和自动求导。

    Parameters
    ----------
    force_body : (num_envs, 3) Tensor
        每个环境的 body 坐标系受力。
    q_body_to_world : (num_envs, 4) Tensor
        每个环境的四元数 [w, x, y, z]，body -> world。

    Returns
    -------
    force_world : (num_envs, 3) Tensor
        世界坐标系下的受力。
    horizontal_mag : (num_envs,) Tensor
        水平方向力幅值。
    """

    w, x, y, z = q_body_to_world[:, 0], q_body_to_world[:, 1], q_body_to_world[:, 2], q_body_to_world[:, 3]
    vx, vy, vz = force_body[:, 0], force_body[:, 1], force_body[:, 2]

    # q * v
    qv_w = -x * vx - y * vy - z * vz
    qv_x = w * vx + y * vz - z * vy
    qv_y = w * vy + z * vx - x * vz
    qv_z = w * vz + x * vy - y * vx

    # (q * v) * q*
    fx = qv_w * -x + qv_x * w + qv_y * z + qv_z * -y
    fy = qv_w * -y + qv_x * -z + qv_y * w + qv_z * x
    fz = qv_w * -z + qv_x * y + qv_y * -x + qv_z * w

    force_world = torch.stack([fx, fy, fz], dim=-1)
    # horizontal_mag = torch.hypot(fx, fy)
    return force_world #, horizontal_mag



def base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def track_lin_vel_exp(
    env: ManagerBasedRLEnv, std_l2: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    error_norm = (env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]).norm(dim=1)
    # 使用 tanh 进行平滑过渡
    smooth_factor = torch.tanh(6.0 * (error_norm - 0.1)) 
    smooth_factor = (smooth_factor + 1.0) / 2.0  
    
    # 死区内：reward=1.0，死区外：使用指数衰减
    exp_reward = torch.exp(-error_norm**2 / std_l2)
    return smooth_factor * exp_reward + (1.0 - smooth_factor) * 1.0
    

def track_ang_vel_exp(
    env: ManagerBasedRLEnv, 
    std_l2: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel with smooth dead zone."""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    error_abs = (env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]).abs()
    smooth_factor = torch.tanh(5. * (error_abs - 0.15))
    smooth_factor = (smooth_factor + 1.0) / 2.0
    
    exp_reward = torch.exp(-error_abs / std_l2)
    
    # 混合奖励
    return smooth_factor * exp_reward + (1.0 - smooth_factor) * 1.0


def tracking_command_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = command[:, 3] + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = command[:, 3]

    # return torch.exp(-torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)/std**2)
    return torch.square((asset.data.root_pos_w[:, 2] - adjusted_target_height).clip(min=0.0, max=0.38))

def lin_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * (env.episode_length_buf > 5)


def flat_orientation(env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ Penalize non-flat base orientation using L2 squared kernel. """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) # * \
        # (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < command_threshold)


def joint_powers_l1(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ Penalize joint powers on the articulation using L1-kernel """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque[:, asset_cfg.joint_ids], asset.data.joint_vel[:, asset_cfg.joint_ids])), dim=1) # * \


def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_link_pos_w[:,asset_cfg.body_ids]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    feet_distance = torch.norm(feet_pos[:, 2, :2] - feet_pos[:, 3, :2], dim=-1)
    reward += torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward


def stand_still(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)# * (torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1) < 0.5)
    

def dof_vel(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(vel), dim=1) * \
        (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < command_threshold)


def feet_contact(env: ManagerBasedRLEnv, 
                 command_name: str,
                 sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.sum(torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 5.0, dim=1)# * \
            # (env.command_manager.get_command(command_name)[:, 2].abs() < 0.1)
    

def joint_deviation(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)# * \
        # (torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1) < 0.087)# * \
        # (env.command_manager.get_command(command_name)[:, 2].abs() < 0.1)


def hip_joint_symmetry(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    joint_ids, _ = asset.find_joints(["LF_ABAD_JOINT", "RF_ABAD_JOINT", "LB_ABAD_JOINT", "RB_ABAD_JOINT"], None, True)
    angle = torch.square(asset.data.joint_pos[:, joint_ids[0]] + asset.data.joint_pos[:, joint_ids[1]]) + torch.square(
                        asset.data.joint_pos[:, joint_ids[2]] + asset.data.joint_pos[:, joint_ids[3]])
    return angle


def feet_stumble(env: ManagerBasedRLEnv, 
                 sensor_cfg: SceneEntityCfg, 
                 sensor_names: list[str],  
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    mask = torch.ones((env.num_envs,), dtype=torch.float32, device=env.device)
    num: int = -int(0.334 * env.num_envs)
    mask[num:] = 0.
    touch = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    for id in sensor_cfg.body_ids:
        quat = asset.data.body_quat_w[:, id]
        force = contact_sensor.data.net_forces_w[:, id, :]
        force_world = force_body_to_world_torch(force, quat)
        touch += (torch.norm(force_world[:, :2], dim=1) > 6.).float() #* torch.abs(force_world[:, 2])).float()
    return touch * mask


def lift_when_blocked(env: ManagerBasedRLEnv, 
                 sensor_cfg: SceneEntityCfg, 
                 sensor_names: list[str],  
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    rew = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    for idx, id in enumerate(sensor_cfg.body_ids):
        sensor: RayCaster = env.scene.sensors[sensor_names[idx]]
        feet_height = (asset.data.body_pos_w[:, id, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1) - 0.26).clip(min=1.e-6, max=0.40)
        quat = asset.data.body_quat_w[:, id]
        force = contact_sensor.data.net_forces_w[:, id, :]
        force_world = force_body_to_world_torch(force, quat)

        rew += (feet_height * (torch.norm(force_world[:, :2], dim=1) >= 5.) * (asset.data.projected_gravity_b[:, 0] < -0.07)).float().clip(min=1.e-6, max=0.6)
    return rew


def tracking_feet_height(env: ManagerBasedRLEnv, 
               sensors: list[str], 
               target_height: float,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
               command_name: str = "commands") -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    phase = env.get_phase()
    lf_rb_swing_mask = (phase < 0.5)
    rf_lb_swing_mask = (phase >= 0.5)

    # target = torch.abs(torch.sin(2 * torch.pi * phase)) * target_height
    lf_sensor: RayCaster = env.scene.sensors[sensors[0]]
    lf_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2] - torch.mean(lf_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    rf_sensor: RayCaster = env.scene.sensors[sensors[1]]
    rf_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2] - torch.mean(rf_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    lb_sensor: RayCaster = env.scene.sensors[sensors[2]]
    lb_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[2], 2] - torch.mean(lb_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    rb_sensor: RayCaster = env.scene.sensors[sensors[3]]
    rb_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[3], 2] - torch.mean(rb_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    
    # lf_rb_phase_reward = torch.exp(-(torch.abs(lf_feet_height - target) + torch.abs(rb_feet_height - target))*100)
    # rf_lb_phase_reward = torch.exp(-(torch.abs(rf_feet_height - target) + torch.abs(lb_feet_height - target))*100)
    lf_rb_phase_reward = torch.clip(lf_feet_height, min=0, max=target_height) + torch.clip(rb_feet_height, min=0, max=target_height)
    rf_lb_phase_reward = torch.clip(rf_feet_height, min=0, max=target_height) + torch.clip(lb_feet_height, min=0, max=target_height)

    return (lf_rb_swing_mask * lf_rb_phase_reward + rf_lb_swing_mask * rf_lb_phase_reward) * \
        (torch.norm(env.command_manager.get_command(command_name)[:, :2]) > 0.1)# * \
        # (env.command_manager.get_command(command_name)[:, 3] > env.command_manager.get_term(command_name).cfg.target_height[0] + 0.05)


def lift_foot(env: ManagerBasedRLEnv, 
               target_pos: float,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
               command_name: str = "commands") -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    phase = env.get_phase()
    lf_rb_swing_mask = (phase < 0.5)
    rf_lb_swing_mask = (phase >= 0.5)
    
    lfrb_target = torch.abs(torch.sin(2 * torch.pi * phase)) * (target_pos + 1.26) - 1.26
    rflb_target = torch.abs(torch.sin(2 * torch.pi * phase)) * (target_pos + 1.30) - 1.30
    lf_rb_err = torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids[:2]] - lfrb_target.unsqueeze(1)), dim=1)
    rf_lb_err = torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids[2:]] - rflb_target.unsqueeze(1)), dim=1)

    lf_rb_phase_reward = torch.exp(-(lf_rb_err*10))
    rf_lb_phase_reward = torch.exp(-(rf_lb_err*10))

    return (lf_rb_swing_mask * lf_rb_phase_reward + rf_lb_swing_mask * rf_lb_phase_reward) * \
        (torch.norm(env.command_manager.get_command(command_name)[:, :2]) > 0.06)


def recovery_thigh(env: ManagerBasedRLEnv, 
               target_pos: float,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]    
    thigh_err = torch.clip((target_pos-asset.data.joint_pos[:, asset_cfg.joint_ids]), min=-5., max=0.)
    return torch.sum(torch.square(thigh_err), dim=1) * (env.episode_length_buf > 5)

    
def reward_stretch(env: ManagerBasedRLEnv, 
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
               command_name: str = "commands") -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    phase = env.get_phase()
    lf_rb_swing_mask = (phase < 0.5)
    rf_lb_swing_mask = (phase >= 0.5)
    lf_rb_stretch_reward = (asset.data.joint_pos[:, asset_cfg.joint_ids[0]] < 0.) & (asset.data.joint_pos[:, asset_cfg.joint_ids[3]] < 0.)
    rf_lb_stretch_reward = (asset.data.joint_pos[:, asset_cfg.joint_ids[1]] < 0.) & (asset.data.joint_pos[:, asset_cfg.joint_ids[2]] < 0.)
    return (lf_rb_swing_mask * lf_rb_stretch_reward.float() + rf_lb_swing_mask * rf_lb_stretch_reward.float()) * \
        (torch.norm(env.command_manager.get_command(command_name)[:, :3]) > 0.08) * \
        (env.command_manager.get_command(command_name)[:, 3] > env.command_manager.get_term(command_name).cfg.target_height[0] + 0.05)


def reward_phase(env: ManagerBasedRLEnv, 
                sensor_cfg: SceneEntityCfg, 
                command_name: str = "commands",
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contact = torch.zeros((env.num_envs, 4),dtype=torch.bool, device=env.device)
    idx = 0
    for id in asset_cfg.body_ids:
        quat = asset.data.body_quat_w[:, id]
        force = sensor.data.net_forces_w[:, id, :]
        force_world = force_body_to_world_torch(force, quat)
        contact[:,idx] = (force_world[: ,2] > 6.)
        idx += 1

    phase = env.get_phase()
    lf_rb_swing_mask = (phase >= 0.5)
    rf_lb_swing_mask = (phase < 0.5)
    has_command = (torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.08)

    trot = ((contact[:,0] == contact[:,3]) & \
        (contact[:,1] == contact[:,2]) & \
        (contact[:,0] == lf_rb_swing_mask) & \
        (contact[:,1] == rf_lb_swing_mask)) & has_command
    stand = contact.all(dim=1) & ~has_command
    return trot.float() + (stand.float() * 2.)


def cp_reward_phase(env: ManagerBasedRLEnv, 
                sensor_cfg: SceneEntityCfg, 
                command_name: str = "commands",
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, asset_cfg.body_ids, 2]

    phase = env.get_phase()
    lf_rb_stance_mask = (phase > 0.5)
    rf_lb_standce_mask = (phase < 0.5)
    has_command = (torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.06)

    lf_rb_trot = (torch.clip(contact[:,0], min=0, max=3) + torch.clip(contact[:,3], min=0, max=3)) * lf_rb_stance_mask
    rf_lb_trot = (torch.clip(contact[:,1], min=0, max=3) + torch.clip(contact[:,2], min=0, max=3)) * rf_lb_standce_mask

    return (lf_rb_trot + rf_lb_trot) * has_command


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty

