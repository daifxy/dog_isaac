# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.sensors import (
    ImuCfg, 
    CameraCfg, 
    ContactSensorCfg, 
    RayCasterCfg, 
    patterns)
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.noise import GaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import UniformNoiseCfg as UniformNoise
from isaaclab.envs.common import ViewerCfg

from . import mdp

##
# Pre-defined configs
##

from dog.assets.robot.config import WL_DOG_CFG  # isort:skip
from dog.assets.robot.config import WL_DOG_ACTION_SCALE  # isort:skip
from dog.assets.robot.config import WL_DOG_LEG_JOINT_NAMES, WL_DOG_WHEEL_JOINT_NAMES  # isort:skip
from dog.assets.terrain.terrain import ROUGH_TERRAIN, PLANE, COMPLETE_ENUE_MESH  # isort:skip

##
# Scene definition
##


@configclass
class DogSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # ground plane
    ground = ROUGH_TERRAIN
    # terrain = COMPLETE_ENUE_MESH

    # robot
    robot: ArticulationCfg = WL_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # 接触传感器
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        force_threshold=1.0,
        history_length=3, 
        update_period=0.0025, 
        track_air_time=True,
        debug_vis=False,
    )
    # 复杂地形中的高度传感器
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        max_distance=10,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.8, 0.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"], 
    )
    lf_foot_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/LF_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.0, -0.0, -0.0)),
        ray_alignment="yaw",
        max_distance=10,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0., 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"], 
    )
    lb_foot_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/LB_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.0, -0.0, -0.0)),
        ray_alignment="yaw",
        max_distance=10,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0., 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"], 
    )
    rf_foot_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/RF_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.0, -0.0, -0.0)),
        ray_alignment="yaw",
        max_distance=10,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0., 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"], 
    )
    rb_foot_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/RB_FOOT_LINK",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.0, -0.0, -0.0)),
        ray_alignment="yaw",
        max_distance=10,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0., 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"], 
    )
    # 深度
    # depth_camera: RayCasterCfg = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #     ray_alignment="base",
    #     max_distance=10,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.8, 0.4], direction=(1., 0., -0.707)),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"], 
    # )
    # 加速度传感器
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0., 0.)),
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=800.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # legs = mdp.JointPositionActionCfg(asset_name="robot", 
    #                                   joint_names=WL_DOG_LEG_JOINT_NAMES, 
    #                                   clip={".*": [-2.7, 2.7]},
    #                                   scale=0.25,
    #                                   preserve_order=True,
    #                                   use_default_offset=True)
    # wheels = mdp.JointVelocityActionCfg(asset_name="robot", 
    #                                     joint_names=WL_DOG_WHEEL_JOINT_NAMES, 
    #                                     clip={".*": [-10.0, 10.0]},
    #                                     scale=5.0,
    #                                     preserve_order=True,
    #                                     use_default_offset=True)
    
    legs = mdp.action_cfg.PosActionCfg(asset_name="robot", 
                                      joint_names=WL_DOG_LEG_JOINT_NAMES, 
                                      clip={".*": [-2.7, 2.7]},
                                      scale=0.25,
                                      preserve_order=True,
                                      use_default_offset=True,
                                      noise=UniformNoise(operation="add", n_min=-0.007, n_max=0.007))
    wheels = mdp.action_cfg.VelActionCfg(asset_name="robot", 
                                        joint_names=WL_DOG_WHEEL_JOINT_NAMES, 
                                        clip={".*": [-10.0, 10.0]},
                                        scale=5.0,
                                        preserve_order=True,
                                        use_default_offset=True,
                                        noise=UniformNoise(operation="add", n_min=-0.005, n_max=0.005))

@configclass
class CommandsCfg:

    # base_velocity = mdp.UniformVelocityCommandCfg( # mdp.terrain_levels_vel() 的实现里使用了 base_velocity，所以这样定义名字
    #     asset_name="robot",
    #     resampling_time_range=(3., 3.),
    #     rel_standing_envs=0.05,
    #     debug_vis=False,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.4, 1.0), lin_vel_y=(-0., 0.), ang_vel_z=(0., 0.), # heading=(-math.pi, math.pi)
    #     ),
    # )

    base_velocity = mdp.CommandCfg( # mdp.terrain_levels_vel() 的实现里使用了 base_velocity，所以这样定义名字
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.12,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.5, 1.5), # heading=(-math.pi, math.pi)
        ),
        probability=0.333,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # base_lin_vel 被encoder计算
        # phase = ObsTerm(func=mdp.get_phase)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(operation="add", n_min=-0.3, n_max=0.3), scale=0.5,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(operation="add", n_min=-0.02, n_max=0.02), scale=1.0,)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                                noise=UniformNoise(operation="add", n_min=-0.06, n_max=0.06),
                                scale=1.0, 
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                                                    joint_names=WL_DOG_LEG_JOINT_NAMES,
                                                                    preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                                noise=UniformNoise(operation="add", n_min=-1.2, n_max=1.2),
                                scale=0.1, 
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                                                    joint_names=[*WL_DOG_LEG_JOINT_NAMES, *WL_DOG_WHEEL_JOINT_NAMES],
                                                                    preserve_order=True)})
        # joint_torque = ObsTerm(func=mdp.robot_joint_torque,
        #                         noise=UniformNoise(operation="add", n_min=-2.6, n_max=2.6),
        #                         scale=0.1, 
        #                         params={"asset_cfg": SceneEntityCfg("robot", 
        #                                                             joint_names=[*WL_DOG_LEG_JOINT_NAMES, *WL_DOG_WHEEL_JOINT_NAMES],
        #                                                             preserve_order=True)})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 4
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        # phase = ObsTerm(func=mdp.get_phase)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,clip=(-100.0, 100.0),scale=0.5,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity,clip=(-100.0, 100.0),scale=1.0,)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                                scale=1.0, 
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                                                    joint_names=WL_DOG_LEG_JOINT_NAMES)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        joint_torque = ObsTerm(func=mdp.robot_joint_torque, scale=0.1,)
        last_action = ObsTerm(func=mdp.last_action)
        # Privileged observation
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0),scale=2.0,) # 必须把需要被encoder估计的量放在从`Privileged observation`开始的前面
        base_acc_vel = ObsTerm(func=mdp.base_lin_acc, 
                               params={"sensor_cfg": SceneEntityCfg(name="imu_sensor")},
                               clip=(-100.0, 100.0),scale=0.05,)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT.*")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        # robot_height = ObsTerm(func=mdp.robot_height, params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("height_scanner")})
        # depth_camera = ObsTerm(func=mdp.depth_camera, params={"sensor_cfg": SceneEntityCfg("depth_camera")})
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_sensor",
                                                                                body_names=[".*"])}
        )

        def __post_init__(self): # rsl里对him的实现不支持critic的历史观测，只能单帧
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class LablesCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0,
                               noise=GaussianNoise(operation="add", mean=0.0, std=0.04))
        # feet_height = ObsTerm(func=mdp.feet_height, clip=(-100.0, 100.0),scale=1.0, 
        #                       params={"sensors": ["lf_foot_height_scanner", "rf_foot_height_scanner", "lb_foot_height_scanner", "rb_foot_height_scanner"],
        #                               "asset_cfg": SceneEntityCfg(name="robot", 
        #                                                           body_names=["lf_foot_link", "rf_foot_link", "lb_foot_link", "rb_foot_link"],
        #                                                           preserve_order=True)})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privileged: CriticCfg = CriticCfg()
    labels: LablesCfg = LablesCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm( # 摩擦力
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 1.0),
            "make_consistent": True,
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm( # 机体质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "distribution": "uniform",
            "mass_distribution_params": (-2.0, 2.5),
            "operation": "add",
        },
    )
    randomize_actuator_gains = EventTerm( # 电机增益
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.99, 1.01),
            "damping_distribution_params": (0.99, 1.01),
            "operation": "scale",
        }
    )
    randomize_physics_scene_gravity = EventTerm( # 重力
        func=mdp.randomize_physics_scene_gravity,
        mode="startup",
        params={
            "gravity_distribution_params": ([-0.01, -0.01, -0.05],[0.01, 0.01, 0.05]), # 9.81
            "operation": "add",
            "distribution": "uniform",
        }
    )
    randomize_base_com_range = EventTerm( # 重心位置
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range":  {"x":(-0.03, 0.03), "y":(-0.03, 0.03), "z":(-0.03, 0.03)}, # 均匀采样，以`add`方式添加到初始值
        }
    )
    randomize_other_com_range = EventTerm( # 其他body重心位置
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_LINK"),
            "com_range":  {"x":(-0.01, 0.01), "y":(-0.01, 0.01), "z":(-0.01, 0.01)},
        }
    )

    # reset
    reset_leg_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*FOOT).*"]),
            "position_range": (-1.5, 1.5),
            "velocity_range": (-0.5, 0.5),
        },
    )
    randomize_spawn_state = EventTerm( # 初始状态
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.0, 0.0],
                "y": [-0.0, 0.0],
                "z": [0.0, 0.0],
                "yaw": [0., 0.],
                "pitch": [-0.25, 0.25],
                "roll": [-0.25, 0.25],
            },
            "velocity_range": {
                "x": [-0.07, 0.07],
                "y": [-0.07, 0.07],
                "z": [-0.07, 0.07],
                "yaw": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "roll": [0.0, 0.0],
            }
        }
    )

    # interval
    # push = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10., 15.),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "velocity_range": {"x": (-1.0, 1.0), 
    #                            "y": (-1.0, 1.0), 
    #                            "z": (-0.2, 0.4), 
    #                            "roll": (-0.5, 0.5), 
    #                            "pitch": (-0.25, 0.25), 
    #                            "yaw": (-0.2, 0.2)},
    #     }
    # )
    wind = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10, 15),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-2.5, 2.5),
            "torque_range": (-0.6, 0.6),
        }
    )
    # slip = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(15., 20.),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT.*"),
    #         "velocity_range": {"x": (-0.4, 0.4), 
    #                            "y": (-0.4, 0.4), 
    #                            "z": (-0.0, 0.2), 
    #                            "roll": (0.0, 0.0), 
    #                            "pitch": (-0.8, 0.8), 
    #                            "yaw": (0.0, 0.0)},
    #     }
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.6, params={"std": 0.632, "command_name": "base_velocity"})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.2, params={"std": 0.632, "command_name": "base_velocity"})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0,
                                params={"threshold": 5.0,
                                        "sensor_cfg": SceneEntityCfg(name="contact_sensor",
                                                                    body_names=["^(?!.*FOOT).*"]),})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z, weight=-0.5)

    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.08)
    torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.00005)
    soft_dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10., 
                                  params={"asset_cfg": SceneEntityCfg("robot", 
                                                                    joint_names=["^(?!.*FOOT).*",])})
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=1.0, params={"command_name": "base_velocity",
                                                                        "threshold": 0.6, 
                                                                        "sensor_cfg": SceneEntityCfg(name="contact_sensor",
                                                                                                     body_names=[".*FOOT.*"]),})
    bad_orientation = RewTerm(func=mdp.flat_orientation, weight=-1.2, params={"command_name":"base_velocity"})
    hip_pos = RewTerm(func=mdp.joint_deviation, weight=-0.1, params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*ABAD.*"), 
                                                                        "command_name": "base_velocity"})
    # hip_symmetry = RewTerm(func=mdp.hip_joint_symmetry, weight=-0.001, params={"command_name": "base_velocity"})
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.005)
    joint_powers_l1 = RewTerm(func=mdp.joint_powers_l1, weight=-2.5e-5, params={"command_name": "base_velocity",
                                                                               "asset_cfg": SceneEntityCfg(name="robot", 
                                                                                                            joint_names=[".*"])})
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.1, 
                          params={"command_name": "base_velocity", 
                                  "command_threshold": 0.1,
                                  "asset_cfg": SceneEntityCfg(name="robot",joint_names=["^(?!.*FOOT).*"],)})
    dof_vel = RewTerm(func=mdp.dof_vel, weight=-0.1, 
                          params={"command_name": "base_velocity", 
                                  "command_threshold": 0.1,
                                  "asset_cfg": SceneEntityCfg(name="robot",joint_names=[".*FOOT.*"],)})
    lift_feet = RewTerm(func=mdp.lift_when_blocked, weight=0.2, 
                           params={"sensor_names": ["lf_foot_height_scanner", "rf_foot_height_scanner", "lb_foot_height_scanner", "rb_foot_height_scanner"],
                                   "sensor_cfg": SceneEntityCfg(name="contact_sensor", 
                                                                body_names=["LF_FOOT_LINK", "RF_FOOT_LINK", "LB_FOOT_LINK", "RB_FOOT_LINK"],
                                                                preserve_order=True),})
    # rew_feet_contact = RewTerm(func=mdp.feet_contact, weight=0.01, params={"command_name": "base_velocity", 
    #                                                                         "sensor_cfg": SceneEntityCfg(name="contact_sensor", 
    #                                                                                                      body_names=[".*FOOT.*"])})             


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_outs = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    terminated = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.65},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # pass

    terrain_levels = CurrTerm(func=mdp.my_terrain_levels_vel)


##
# Environment configuration
##


@configclass
class DogEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DogSceneCfg = DogSceneCfg(num_envs=2048, env_spacing=0., filter_collisions=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 8
        self.episode_length_s = 10
        # self.ui_window_class_type = None
        # viewer settings
        self.viewer = ViewerCfg(
            resolution=(1280, 1024),
            eye = (3.0, 3.0, 1.5),
            lookat = (0., 0., 0.),
            origin_type = "asset_root",
            asset_name="robot",
        )
        # simulation settings
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.only_positive_rewards = True

        self.cycle_length = 1.2