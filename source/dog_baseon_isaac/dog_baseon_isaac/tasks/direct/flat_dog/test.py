from dog_baseon_isaac.assets.robots.dog import DOG_CFG
from dog_baseon_isaac.assets.terrains.terrains import ROUGH_TERRAINS_GENERATOR_CFG

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
import isaaclab.envs.ui as ui
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, GroundPlaneCfg, materials
from isaaclab.utils import configclass, string
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModel, NoiseModelCfg

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, ImuCfg, CameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import math

# -- 域随机化配置
@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTermCfg( # 随机化物理材质-摩擦力
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("Robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "make_consistent": False,
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTermCfg( # 随机化质量-机体质量
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("Robot", body_names="base_link"),
            "mass_distribution_params": (0.96, 1.04),
            "operation": "scale",
        },
    )

    randomize_actuator_gains = EventTermCfg( # 随机化电机增益
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("Robot", body_names=".*"),
            "stiffness_distribution_params": (0.96, 1.04),
            "damping_distribution_params": (0.96, 1.04),
            "operation": "scale",
        }
    )


def get_env_cfg():

    joints_names = [
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    ]
    joint_pos_limits = {   # [rad]
        "FL_hip_joint": [-0.62, 0.62],
        "FR_hip_joint": [-0.62, 0.62],
        "RL_hip_joint": [-0.62, 0.62],
        "RR_hip_joint": [-0.62, 0.62],
        "FL_thigh_joint": [-2.0, 1.0],
        "FR_thigh_joint": [-2.0, 1.0],
        "RL_thigh_joint": [-2.0, 1.0],
        "RR_thigh_joint": [-2.0, 1.0],
        "FL_calf_joint": [0, 2.0],
        "FR_calf_joint": [0, 2.0],
        "RL_calf_joint": [0, 2.0],
        "RR_calf_joint": [0, 2.0],
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0], 
        "ang_vel_range": [-1.5, 1.5],
        "half_length_of_body": 0.16,
    }
    curriculum_cfg = {
        "curriculum_commands_check_interval": 280,  #每多少step更新一次课程学习
        "curriculum_level_growth_rate":0.005, 
        "curriculum_min_commands_proportion":0.3, 
        "err_mode": True, #误差模式 True  动态误差模式 False  要注意在地形情况下速度跟踪肯定和平地差距很大注意误差调整
        "lin_vel_err_range":[0.35,0.38],  #[0.35,0.5]  
        "ang_vel_err_range":[0.5,0.56],  #[0.5,1.0]    
        "joint_max_vel_limit":25
    }
    other_randomize_cfg = { # 手动调用，在这里配置参数
        "randomize_base_com_range":  {"x":(-0.015, 0.015), "y":(-0.015, 0.015), "z":(-0.02, 0.02)}, # 均匀采样，以`add`方式添加到初始值
        "randomize_other_com_range": {"x":(-0.005, 0.005), "y":(-0.005, 0.005), "z":(-0.01, 0.01)}, 
        "randomize_physics_scene_gravity": {
            "gravity_distribution_params": ([-0.01, -0.01, -0.05],[0.01, 0.01, 0.05]), # 9.81
            "operation": "add",
            "distribution": "uniform",
        },
        "randomize_rigid_body_collider_offsets": {
            "rest_offset_distribution_params": (0.0, 0.008),
            "contact_offset_distribution_params": (0.0, 0.01), 
            "distribution": "uniform" , # 目前支持绝对值的赋值方式:“abs”
        },
        "external_force_and_torque" : { # 随机施加外力
            "enabled": True,
            "applied_bodies": ["base_link"],
            "max_force": 2.5,
            "max_torque": 1.0,
            "duration": [0.1, 0.5], # s
            "position": [0.14, 0.14, 0.14],
            "interval": 10, # s
            "is_global": False,
            "curriculum_force_check_interval": 210,
            "curriculum_force_level_growth_rate":0.01,
            "curriculum_min_force_proportion": 0.2,
        },
        "randomize_respawn_point": { # 随机量在`data.default_root_state`的基础上直接相加，而不改变`data.default_root_state`的值
            "pose_range": {
                "x": [-0.25, 0.25],
                "y": [-0.25, 0.25],
                "z": [0.0, 0.1],
                "yaw": [-math.pi, math.pi],
                "pitch": [-0.45, 0.45],
                "roll": [-0.45, 0.45],
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
    }
    env_cfg = {
        "command_space" : 3, 
        "history_length" : 3,
        "action_space" : 12,
        "policy_slice_obs": 36,
        "observation_space" : 159, # 36*4 + last_action(12) + command(3)
        "state_space" : 160, # 159 + privileged_obs(1)
        
        "dt" : 0.005,
        "decimation": 4,
        "resample_commands_s" : 10, # s
        "episode_length_s" : 32,

        "log_dir": None, # None：默认
        "termination_if_base_connect_plane": False, 
        "die_if_contact_bodies": "base_link",
        "undesired_contact_bodies":[
            ".*_hip_Link",
            ".*_thigh_Link",
        ],
        "feet_contact_bodies":[
            ".*_calf_Link",
        ],
        "enable_terrain": False,
    }
    noise_cfg = {
        "noise": True,
        "noise_level": 1.0,
        "dead_zone": 0.05,
        "noise_scales":{
            "lin_acc": 0.01,
            "lin_vel": 0.1,
            "ang_vel": 0.2,
            "dof_pos": 0.01,
            "dof_vel": 0.8,
            "gravity": 0.05
        }
    }
    obs_scales = {
        "action_scale" : 0.25,
        "clip_actions" : 100.0,
        "clip_observations" : 100.0,

        "lin_vel": 2.0,
        "ang_vel": 2.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,
        "lin_acc": 0.1,
    }
    reward_cfg = {
        # TODO z轴加大，dof_vel修改
        "only_positive_rewards" : True, 
        "reward_offset": 1.,
        "tracking_lin_vel_sigma": 0.25, 
        "tracking_ang_sigma": 0.25, 
        "reward_scales": {
            "termination": -0.5,
            "tracking_lin_xy_vel": 1.4,
            "tracking_ang_vel": 0.6,
            "lin_vel_z": -2.5, 
            # "tracking_height": -16,
            "dof_acc": -2.5e-7,
            "base_ang_vel": -0.05,
            "action_rate":-0.01,
            # "survive": 1.0,
            # "default_hip": -0.4,
            # "same_side_hip_similar":-0.12, 
            # "diagonal_thigh_calf_similar": -0.06,
            # "orientation": -0.1,
            "feet_air_time": 1.,
            # "stand_steadily": -10,
            # "stable_feet": 0.06,
            "undesired_contacts": -1.0,
            "joint_torques": -0.0001,
            "stand_still": -0.2,
            # "no_jump": -0.1,
            # "joint_vel": -0., # -6e-3
            # "feet_force": -1e-4,
        },
    }

    return joints_names, joint_pos_limits, env_cfg, noise_cfg, obs_scales, command_cfg, reward_cfg, curriculum_cfg, other_randomize_cfg

ROUGH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_GENERATOR_CFG,
    max_init_terrain_level=9,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        project_uvw=True,
    ),
    debug_vis=False,
)

PLANE = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    debug_vis=False,
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    enable_terrain = False
    terrain: TerrainImporterCfg = ROUGH_TERRAIN_CFG if enable_terrain else PLANE

    # robot(s)
    robot_cfg: ArticulationCfg = DOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensors
    # 复杂地形中的高度传感器
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.0685, -0.0250, -0.0649)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.1, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 接触传感器
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        force_threshold=1.0,
        history_length=3, 
        update_period=0.005, 
        track_air_time=True,
        debug_vis=False,
    )
    # 加速度传感器
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.0685, -0.0250, -0.0649)),
        debug_vis=False,
    )
    # 相机传感器
    camera_sensor: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/front_cam",
        offset=CameraCfg.OffsetCfg(pos=(0.0685, -0.0250, -0.0649), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(clipping_range = (0.01, 1e4),),
        width=64,
        height=48,
        debug_vis=False,
    )

    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


@configclass
class DogEnvCfg(DirectRLEnvCfg):
    
    joints_names, joint_pos_limits, env_cfg, noise_cfg, obs_scales, command_cfg, reward_cfg, curriculum_cfg, other_randomize_cfg = get_env_cfg()
    # -------------------------------------------------------------------------
    viewer = ViewerCfg(
        eye= (5., 5., 5.),
        lookat= (0., 0., 0.),
        resolution= (1280, 720),
        origin_type= "asset_body",
        env_index= 0,
        asset_name= "Robot",
        body_name= "base_link",
    )         
    # ui_window_class_type = ui.BaseEnvWindow
    seed = 42            
    is_finite_horizon = False    # 有/无限时域 False: 无限时域
    rerender_on_reset = False
    # -------------------------------------------------------------------------

    # 域随机化事件
    events:EventCfg = EventCfg()
        
    decimation = env_cfg["decimation"]
    dt=env_cfg["dt"]
    resample_commands_s = env_cfg["resample_commands_s"]
    episode_length_s = env_cfg["episode_length_s"]
    action_scale = obs_scales["action_scale"]
    clip_actions = obs_scales["clip_actions"]
    clip_observations = obs_scales["clip_observations"]
    # - spaces definition
    command_space = env_cfg["command_space"]
    action_space = env_cfg["action_space"]
    observation_space = env_cfg["observation_space"]
    state_space = env_cfg["state_space"]


    # log_dir 
    log_dir = env_cfg["log_dir"] 
    
# ----------------------------------------------------------------------------------------------------------------------
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation, device="cuda:0")
    # scene
    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


    @property
    def resample_commands_length(self) -> int:
        return math.ceil(self.resample_commands_s / (self.sim.dt * self.decimation))
