from dog_baseon_isaac.assets.robots.dog import DOG_CFG
from dog_baseon_isaac.assets.terrains.terrains import ROUGH_TERRAINS_CFG, PLANE

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, GroundPlaneCfg, materials
from isaaclab.utils import configclass, string
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModel, NoiseModelCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

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
            "static_friction_range": (0.6, 1.4),
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
        "lin_vel_x_range": [-1.0, 1.0], #修改范围要调整奖励权重 低速范围约[-1.0,1.0]
        "lin_vel_y_range": [-0.5, 0.5], 
        "ang_vel_range": [-3.14, 3.14],   #修改范围要调整奖励权重 低速范围约[-3.14,3.14]
        "height":[0.25, 0.33],    #高度范围
        "half_length_of_body": 0.16,
        "inverse_linx_angv": 1,    #前进速度和角速度反比 angv <= inverse_linx_angv / linv_x (desmos函数图像:y=\ \frac{i}{x}\left\{-10<y<10\right\}\left\{-2<x<2\right\})
        "inverse_liny_angv": 1,    
    }
    curriculum_cfg = {
        "curriculum_commands_check_interval": 280,  #每多少step更新一次课程学习
        "curriculum_level_growth_rate":0.005,   #比例    0.001
        "curriculum_min_commands_proportion":0.3,   #比例 
        "err_mode": False, #误差模式 True  动态误差模式 False  要注意在地形情况下速度跟踪肯定和平地差距很大注意误差调整
        "lin_vel_err_range":[0.35,0.38],  #[0.35,0.5]  课程误差阈值(上升/下降) 误差 or 误差比例(上升/下降)上升阈值会在前期从下降阈值误差下降到设定值
        "ang_vel_err_range":[0.5,0.56],  #[0.5,1.0]    
        "joint_max_vel_limit":25
    }
    other_randomize_cfg = { # 手动调用，在这里配置参数
        "randomize_base_com_range":  {"x":(-0.035, 0.035), "y":(-0.035, 0.035), "z":(-0.035, 0.035)}, # 均匀采样，以`add`方式添加到初始值
        "randomize_other_com_range": {"x":(-0.01, 0.01), "y":(-0.01, 0.01), "z":(-0.01, 0.01)},
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
            "enabled": False,
            "applied_bodies": ["base_link"],
            "max_force": 1.8,
            "max_torque": 1.0,
            "duration": [0.3, 0.8], # s
            "position": [0.14, 0.14, 0.14],
            "interval": 3, # s
            "is_global": False,
            "curriculum_force_check_interval": 210,
            "curriculum_force_level_growth_rate":0.005,
            "curriculum_min_force_proportion": 0.25,
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
        "observation_cfg" : {
                        #3          12          12              3                 12          3
            "actor": ["ang_vel", "joint_pos", "joint_vel", "projected_gravity", "actions"], # 3+12+12+3+4+12+3 = 42
            "critic": ["actor_obs", "lin_vel", "height"], # 45 + 3 + 1 = 49
        },
        "command_space" : 3, # 7
        "history_length" : 5,
        "action_space" : 12,
        "policy_obs": 42,
        "observation_space" : 255, # 42*6 +3 
        "state_space" : 259, # 57
        
        "num_envs": 2048,
        "dt" : 0.005,
        "decimation" : 4, # 控制频率为60hz 控制频率 = 1 /（dt * decimation）
        "resample_commands_s" : 8, # 8s
        "episode_length_s" : 32,
        "action_scale" : 0.5,
        "clip_actions" : 100.0,
        "obs_noise": True,
        "action_noise": True,
        "log_dir": None, # None：不设置，按照默认
        "termination_if_base_connect_plane": False, #触地重置
        "die_if_contact_bodies": "base_link", #触地死亡
        "undesired_contact_bodies":[
            ".*_hip_Link",
            ".*_thigh_Link",
        ],
        "feet_contact_bodies":[
            ".*_calf_Link",
        ],
        "enable_terrain": False,
    }
    reward_cfg = {
        # TODO 设置课程学习，不同关节速度限制
        "tracking_lin_vel_sigma": 0.25, 
        "tracking_ang_sigma": 0.25, 
        "reward_scales": {
            "tracking_lin_xy_vel": 1.5,
            "tracking_ang_vel": 1.5,
            "lin_vel_z": -1.4, 
            "tracking_height": -14.2,
            "dof_acc": -3e-8,
            "base_ang_vel": -0.05,
            "action_rate":-0.01,
            "survive": 0.7,
            "default_hip": -0.63,
            "same_side_hip_similar":-0.12, 
            "diagonal_thigh_calf_similar": -0.15,
            "projected_gravity": -2.0,
            "feet_air_time": 0.5,
            # "stand_steadily": -10,
            "undesired_contacts": -3.0,
            "joint_torques": -1.5e-4,
            "joint_vel": -5e-3,
            "feet_force": -1e-4,
        },
    }

    return joints_names, joint_pos_limits, env_cfg, command_cfg, reward_cfg, curriculum_cfg, other_randomize_cfg

@configclass
class DogEnvCfg(DirectRLEnvCfg):
    
    joints_names, joint_pos_limits, env_cfg, command_cfg, reward_cfg, curriculum_cfg, other_randomize_cfg = get_env_cfg()
    # -------------------------------------------------------------------------
    viewer = ViewerCfg()         
    ui_window_class_type = None  
    seed = 42                    # 随机种子
    is_finite_horizon = False    # 有/无限时域 False: 无限时域
    rerender_on_reset = False
    # -------------------------------------------------------------------------

    # 域随机化事件
    events:EventCfg = EventCfg()

    # 输入输出噪声
    if env_cfg["obs_noise"]:
        observation_noise_model: NoiseModelCfg = NoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.008,operation="add")
        )
    if env_cfg["action_noise"]:
        action_noise_model: NoiseModelCfg = NoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.008,operation="add")
        )
    
    decimation = env_cfg["decimation"]
    dt=env_cfg["dt"]
    resample_commands_s = env_cfg["resample_commands_s"]
    episode_length_s = env_cfg["episode_length_s"]
    action_scale = env_cfg["action_scale"]
    clip_actions = env_cfg["clip_actions"]
    # - spaces definition
    command_space = env_cfg["command_space"]
    action_space = env_cfg["action_space"]
    observation_space = env_cfg["observation_space"]
    state_space = env_cfg["state_space"]

    # 地形
    terrain: TerrainImporterCfg = None
    if env_cfg["enable_terrain"]:
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
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
    else:
        terrain = TerrainImporterCfg(
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
        history_length=3, update_period=0.005, 
        track_air_time=True,
        debug_vis=False,
    )

    # log_dir 
    log_dir = env_cfg["log_dir"] 
    
# ----------------------------------------------------------------------------------------------------------------------
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation, device="cuda:0")
    # robot(s)
    robot_cfg: ArticulationCfg = DOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=env_cfg["num_envs"], env_spacing=4.0, replicate_physics=True)
    
    @property
    def resample_commands_length(self) -> int:
        return math.ceil(self.resample_commands_s / (self.sim.dt * self.decimation))
