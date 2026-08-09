from __future__ import annotations

from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, RayCaster, Imu

import torch


def wheel_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    Keep the joint position of the wheel between -2pi and 2pi.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    change = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    asset.data.default_root_state
    return torch.fmod(change, 2 * torch.pi)


def robot_joint_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque


def robot_joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc


def feet_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_tensor = contact_sensor.data.net_forces_w_history
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_mass.to(env.device)


def robot_inertia(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    inertia_tensor = asset.data.default_inertia
    return inertia_tensor.view(inertia_tensor.shape[0], -1).to(env.device)


def robot_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_joint_pos


def robot_joint_stiffness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_joint_stiffness


def robot_joint_damping(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_joint_damping


def robot_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def robot_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_vel_w


def robot_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    phase = env.get_phase()
    sin = torch.sin(2 * torch.pi * phase)
    cos = torch.cos(2 * torch.pi * phase)
    return torch.stack([sin, cos], dim=-1)


def base_lin_acc(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    imu: Imu = env.scene.sensors[sensor_cfg.name]
    return imu.data.lin_acc_b

# def stairs_msg(env: ManagerBasedRLEnv, 
#                stairs_max_height: list[str], 
#                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]

#     msg = torch.zeros((env.num_envs, 2), torch.float32, device=env.device)
#     msg 
#     terrain = env.scene.terrain
#     ids = 0.66 * env.num_envs
#     msg[:ids, 0] = terrain.terrain_levels[:ids] * stairs_max_height



    # lf_sensor: RayCaster = env.scene.sensors[sensors[0]]
    # lf_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2] - torch.mean(lf_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    # rf_sensor: RayCaster = env.scene.sensors[sensors[1]]
    # rf_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2] - torch.mean(rf_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    # lb_sensor: RayCaster = env.scene.sensors[sensors[2]]
    # lb_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[2], 2] - torch.mean(lb_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    # rb_sensor: RayCaster = env.scene.sensors[sensors[3]]
    # rb_feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids[3], 2] - torch.mean(rb_sensor.data.ray_hits_w[..., 2], dim=1) -0.02
    
    # return torch.stack([lf_feet_height, rf_feet_height, lb_feet_height, rb_feet_height], dim=-1)
