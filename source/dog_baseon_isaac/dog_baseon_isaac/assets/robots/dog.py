import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg, DCMotorCfg, ImplicitActuatorCfg
import os
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

USD_PATH = os.path.dirname(__file__)

ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*"],
    min_delay=0,
    max_delay=2,
    effort_limit=10, 
    effort_limit_sim=8, 
    velocity_limit=26, # 7.5
    velocity_limit_sim=25, # 7
    stiffness={".*": 30.0},
    damping={".*": 1.0},
    armature={".*": 0.002},
    friction={".*": 0.09},
    dynamic_friction={".*": 0.08},
)

DOG_CFG = ArticulationCfg(
    prim_path = "/World/envs/env_.*/Robot",
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/usd/dog/dog.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=500.0,
            max_angular_velocity=500.0,
            max_depenetration_velocity=5.0,
            ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=4,
            fix_root_link=False,
            ),
        # contact_offset:碰撞形状的接触偏移（以米为单位）。当两个形状的距离小于它们接触偏移之和时，碰撞检测器会生成接触点。
        #                该数值应为非负，这意味着接触生成可能在形状实际穿透之前就开始。
        # rest_offset:碰撞形状的静止偏移（以米为单位）。静止偏移量衡量形状在静止时与其他形状接近的程度。
        #             在静止状态下，两个垂直堆叠的物体之间的距离是它们静止偏移量的总和。
        #             如果一对形状具有正的静止偏移量，这些形状在静止时将通过空气间隙分开。
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.004, rest_offset=0.004, torsional_patch_radius=0.03, min_torsional_patch_radius=0.005
            ),
        joint_drive_props = sim_utils.JointDrivePropertiesCfg(
            drive_type="force", max_effort=8, max_velocity=25
            ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
            # "FL_hip_joint":   0.,  
            # "FR_hip_joint":   0., 
            # "RL_hip_joint":   0.,  
            # "RR_hip_joint":   0.,  
            # ".*_thigh_joint": -0.22,  
            # "FL_calf_joint":  1.25,  
            # "FR_calf_joint":  1.25,  
            # "RL_calf_joint":  1.2,  
            # "RR_calf_joint":  1.2,  
            "FL_hip_joint":   0.,  
            "FR_hip_joint":   0., 
            "RL_hip_joint":   0.,  
            "RR_hip_joint":   0.,  
            "FL_thigh_joint": -0.38,  
            "FR_thigh_joint": -0.38,  
            "RL_thigh_joint": -0.25,  
            "RR_thigh_joint": -0.25,  
            "FL_calf_joint":  1.35,  
            "FR_calf_joint":  1.35,  
            "RL_calf_joint":  1.3,  
            "RR_calf_joint":  1.3,  
        },
        joint_vel={".*": 0.0},
        pos=(0.0, 0.0, 0.40),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    
    actuators = {"joints": ACTUATOR_CFG},
    soft_joint_pos_limit_factor = 0.9,
)

