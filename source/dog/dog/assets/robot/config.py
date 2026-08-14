import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg, DCMotorCfg, ImplicitActuatorCfg
import os
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

USD_PATH = os.path.dirname(__file__)

DOG_12DOF_JOINT_NAMES = [
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

DOG_12DOF_CFG = ArticulationCfg(
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
            enabled_self_collisions=False, # 自碰撞
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=4,
            fix_root_link=False,
            ),
        # contact_offset:碰撞形状的接触偏移（以米为单位）。当两个形状的距离小于它们接触偏移之和时，碰撞检测器会生成接触点。
        # rest_offset:碰撞形状的静止偏移（以米为单位）。静止偏移量衡量形状在静止时与其他形状接近的程度。
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01, rest_offset=0.0, torsional_patch_radius=0.03, min_torsional_patch_radius=0.005
            ),
        joint_drive_props = sim_utils.JointDrivePropertiesCfg(
            drive_type="force", max_effort=8, max_velocity=25
            ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
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
    
    actuators = {
        "joints": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            min_delay=0,
            max_delay=2,
            effort_limit={
                ".*": 10
                }, 
            effort_limit_sim={
                ".*": 8
            }, 
            velocity_limit={
                ".*": 26
            },
            velocity_limit_sim={
                ".*": 25
            },
            stiffness={
                ".*": 30.0
            },
            damping={
                ".*": 1.0
            },
            armature={
                ".*": 0.01
            },
            friction={
                ".*": 0.2
            },
            # dynamic_friction={".*": 0.08},
        )
    },
    soft_joint_pos_limit_factor = 0.9,
)


DOG_12DOF_ACTION_SCALE = {}
for a in DOG_12DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            DOG_12DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


A1_CFG = ArticulationCfg(
    prim_path = "/World/envs/env_.*/Robot",
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/usd/unitree_a1/a1.usd",
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
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01, rest_offset=0.0, torsional_patch_radius=0.03, min_torsional_patch_radius=0.005
            ),
        joint_drive_props = sim_utils.JointDrivePropertiesCfg(
            drive_type="force", max_effort=8, max_velocity=25
            ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
            'FL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.8,   # [rad]
            'FR_calf_joint': -1.8,  # [rad]
            'RL_calf_joint': -1.8,    # [rad]
            'RR_calf_joint': -1.8,    # [rad]
        },
        joint_vel={".*": 0.0},
        pos=(0.0, 0.0, 0.40),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    
    actuators = {
        "joints": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            min_delay=0,
            max_delay=2,
            effort_limit={
                ".*": 10
                }, 
            effort_limit_sim={
                ".*": 8
            }, 
            velocity_limit={
                ".*": 26
            },
            velocity_limit_sim={
                ".*": 25
            },
            stiffness={
                ".*": 30.0
            },
            damping={
                ".*": 1.0
            },
            armature={
                ".*": 0.01
            },
            friction={
                ".*": 0.2
            },
            # dynamic_friction={".*": 0.08},
        )
    },
    soft_joint_pos_limit_factor = 0.9,
)




RCDOG_12DOF_JOINT_NAMES = [
    "lf_hip_roll_joint",  
    "lf_hip_pitch_joint",  
    "lf_knee_joint",  
    "rf_hip_roll_joint", 
    "rf_hip_pitch_joint",  
    "rf_knee_joint",  
    "lb_hip_roll_joint",  
    "lb_hip_pitch_joint",  
    "lb_knee_joint",  
    "rb_hip_roll_joint",  
    "rb_hip_pitch_joint",  
    "rb_knee_joint",                  
]


RCDOG_12DOF_CFG = ArticulationCfg(
    prim_path = "/World/envs/env_.*/Robot",
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/usd/rcdog/rcdog.usd",
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
            enabled_self_collisions=False, # 自碰撞
            solver_position_iteration_count=6, 
            solver_velocity_iteration_count=2,
            fix_root_link=False,
            ),
        # contact_offset:碰撞形状的接触偏移（以米为单位）。当两个形状的距离小于它们接触偏移之和时，碰撞检测器会生成接触点。
        # rest_offset:碰撞形状的静止偏移（以米为单位）。静止偏移量衡量形状在静止时与其他形状接近的程度。
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01, rest_offset=0.0, torsional_patch_radius=0.03, min_torsional_patch_radius=0.005
            ),
        joint_drive_props = sim_utils.JointDrivePropertiesCfg(
            drive_type="force", max_effort=42, max_velocity=376
            ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "lf_hip_roll_joint": 0.,  
            "rf_hip_roll_joint": 0., 
            "lb_hip_roll_joint": 0.,  
            "rb_hip_roll_joint": 0.,  
            "lf_hip_pitch_joint": 0.75,  
            "rf_hip_pitch_joint": 0.75,  
            "lb_hip_pitch_joint": 0.75,  
            "rb_hip_pitch_joint": 0.75,  
            "lf_knee_joint": -1.40,  
            "rf_knee_joint": -1.40,  
            "lb_knee_joint": -1.40,  
            "rb_knee_joint": -1.40,                  
        },
        joint_vel={".*": 0.0},
        pos=(0., -0., 0.36),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    
    actuators = {
        "joints": DelayedPDActuatorCfg(
            joint_names_expr=RCDOG_12DOF_JOINT_NAMES,
            min_delay=0,
            max_delay=1,
            effort_limit=30., 
            effort_limit_sim=29.5, 
            velocity_limit=376.,
            velocity_limit_sim=375.5,
            stiffness=100.,
            damping=5.,
            armature=0.02,
            friction=0.02,
            dynamic_friction=0.01,
        )
    },
    soft_joint_pos_limit_factor = 0.9,
)


RCDOG_12DOF_ACTION_SCALE = {}
for a in RCDOG_12DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            RCDOG_12DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]




WL_DOG_LEG_JOINT_NAMES = [
    "LF_ABAD_JOINT",  
    "LF_HIP_JOINT",  
    "LF_KENN_JOINT",  
    "RF_ABAD_JOINT",  
    "RF_HIP_JOINT",  
    "RF_KENN_JOINT",  
    "LB_ABAD_JOINT",  
    "LB_HIP_JOINT",  
    "LB_KENN_JOINT",  
    "RB_ABAD_JOINT",
    "RB_HIP_JOINT",
    "RB_KENN_JOINT",
]

WL_DOG_WHEEL_JOINT_NAMES = [
    "LF_FOOT_JOINT",    
    "RF_FOOT_JOINT",  
    "LB_FOOT_JOINT",  
    "RB_FOOT_JOINT",  
]

WL_DOG_LEG_BODY_NAMES = [
    "LF_ABAD_LINK",  
    "LF_HIP_LINK",  
    "LF_KENN_LINK",  
    "RF_ABAD_LINK",  
    "RF_HIP_LINK",  
    "RF_KENN_LINK",  
    "LB_ABAD_LINK",  
    "LB_HIP_LINK",  
    "LB_KENN_LINK",  
    "RB_ABAD_LINK",
    "RB_HIP_LINK",
    "RB_KENN_LINK",
]

WL_DOG_WHEEL_BODY_NAMES = [
    "LF_FOOT_LINK",    
    "RF_FOOT_LINK",  
    "LB_FOOT_LINK",  
    "RB_FOOT_LINK",  
]


WL_DOG_CFG = ArticulationCfg(
    prim_path = "/World/envs/env_.*/Robot",
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/usd/wl_dog/wl_dog.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=500.0,
            max_angular_velocity=500.0,
            max_depenetration_velocity=10.0,
            sleep_threshold=0.004
            ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, # 自碰撞
            solver_position_iteration_count=6, 
            solver_velocity_iteration_count=2,
            fix_root_link=False,
            ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.004, rest_offset=0.001, torsional_patch_radius=0.08, min_torsional_patch_radius=0.005
            ),
        joint_drive_props = sim_utils.JointDrivePropertiesCfg(
            drive_type="force", max_effort=42, max_velocity=376
            ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "LF_ABAD_JOINT": 0.,  
            "RF_ABAD_JOINT": 0., 
            "LB_ABAD_JOINT": 0.,  
            "RB_ABAD_JOINT": 0.,  
            "LF_HIP_JOINT": 0.55,  
            "RF_HIP_JOINT": 0.55,  
            "LB_HIP_JOINT": 0.76,  
            "RB_HIP_JOINT": 0.76,  
            "LF_KENN_JOINT": -1.6,  
            "RF_KENN_JOINT": -1.6,  
            "LB_KENN_JOINT": -1.68,  
            "RB_KENN_JOINT": -1.68,                  
        },
        joint_vel={".*": 0.0},
        pos=(0., -0., 0.42),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    
    actuators = {
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=WL_DOG_LEG_JOINT_NAMES,
            min_delay=0,
            max_delay=1,
            effort_limit=36., 
            effort_limit_sim=36., 
            velocity_limit=24.,
            velocity_limit_sim=24.,
            stiffness=80.,
            damping=4.,
            armature=0.01,
            # friction=0.1,
            # dynamic_friction=0.01,
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=WL_DOG_WHEEL_JOINT_NAMES,
            min_delay=0,
            max_delay=1,
            effort_limit=20., 
            effort_limit_sim=20., 
            velocity_limit=43.,
            velocity_limit_sim=42.9,
            stiffness=0.,
            damping=4.,
            armature=0.01,
            # friction=0.1,
            # dynamic_friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor = 0.9,
)


WL_DOG_ACTION_SCALE = {}
for a in WL_DOG_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            WL_DOG_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
