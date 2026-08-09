import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
from collections import deque
from pynput import keyboard
import argparse

# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path('/home/rp/dog/source/dog/dog/assets/robot/mjcf/wl_dog/wl_dog.xml')
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DECIMATION = 8
TIME_STEP = m.opt.timestep * DECIMATION
# FPS = 60.

OBS_HISTORY_KEYS = [
    "velocity_commands",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "joint_frc",
    "actions",
]

MJ_JOINT_NAMES = [
    "LF_ABAD_JOINT",  
    "LF_HIP_JOINT",  
    "LF_KENN_JOINT", 
    "LF_FOOT_JOINT", 
    "RF_ABAD_JOINT", 
    "RF_HIP_JOINT",  
    "RF_KENN_JOINT",
    "RF_FOOT_JOINT",  
    "LB_ABAD_JOINT",  
    "LB_HIP_JOINT",  
    "LB_KENN_JOINT", 
    "LB_FOOT_JOINT", 
    "RB_ABAD_JOINT",  
    "RB_HIP_JOINT",  
    "RB_KENN_JOINT",  
    "RB_FOOT_JOINT",                
]

LAB_LEG_JOINT_NAMES = [
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

LAB_WHEEL_JOINT_NAMES = [
    "LF_FOOT_JOINT",    
    "RF_FOOT_JOINT",  
    "LB_FOOT_JOINT",  
    "RB_FOOT_JOINT",  
]

LAB_JOINT_NAMES = [*LAB_LEG_JOINT_NAMES, *LAB_WHEEL_JOINT_NAMES]


class RoboCfg:

    stiffness = np.array([
        100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 0., 0., 0., 0.
    ], dtype=np.float32)
         
    damping = np.array([
        5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 3., 3., 3., 3.
    ], dtype=np.float32)

    default_pos = np.array([
        0., 0.86, -1.42, 0., 0.86, -1.42, 0., 0.76, -1.42, 0., 0.76, -1.42, 0., 0., 0., 0.
    ], dtype=np.float32)

    tau_limit = np.array([
        36., 36., 36., 36., 36., 36., 36., 36., 36., 36., 36., 36., 20., 20., 20., 20.
    ], dtype=np.float32)

    action_scale = np.array([
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 5., 5., 5., 5.
    ], dtype=np.float32)

    action_clip = np.array([
        2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 10., 10., 10., 10.
    ], dtype=np.float32)
    ''' 以上皆基于Lab关节顺序 '''

    imu_quat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    imu_gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    pos_sensor_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name + "_p") for name in LAB_JOINT_NAMES]
    vel_sensor_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name + "_v") for name in LAB_JOINT_NAMES]
    frc_sensor_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name + "_f") for name in LAB_JOINT_NAMES]

    usd2urdf = [MJ_JOINT_NAMES.index(name) for name in LAB_JOINT_NAMES]
    frame_stack = 5
    num_single_obs = 69         
    num_observations = num_single_obs * frame_stack  # 345
    num_actions = 16


class History:

    is_first_frame = True
    class TermHistory:
        """Maxlen ring buffer, flattened oldest-to-newest per observation term."""

        def __init__(self, max_len: int, term_dim: int):
            self.max_len = max_len
            self.term_dim = term_dim
            self._dq: deque[np.ndarray] = deque(maxlen=max_len)

        def reset(self):
            self._dq.clear()

        def append(self, x: np.ndarray):
            self._dq.append(np.asarray(x, dtype=np.float32).reshape(-1))

        def fill_tile(self, x: np.ndarray):
            self.reset()
            v = np.asarray(x, dtype=np.float32).reshape(-1)
            for _ in range(self.max_len):
                self._dq.append(v.copy())

        def flat(self) -> np.ndarray:
            if len(self._dq) == 0:
                return np.zeros(self.max_len * self.term_dim, dtype=np.float32)
            return np.concatenate(list(self._dq), axis=0)

    obs_buffer = {
        "base_ang_vel": TermHistory(RoboCfg.frame_stack, 3),
        "projected_gravity": TermHistory(RoboCfg.frame_stack, 3),
        "velocity_commands": TermHistory(RoboCfg.frame_stack, 3),
        "joint_pos": TermHistory(RoboCfg.frame_stack, RoboCfg.num_actions - 4),
        "joint_vel": TermHistory(RoboCfg.frame_stack, RoboCfg.num_actions),
        "joint_frc": TermHistory(RoboCfg.frame_stack, RoboCfg.num_actions),
        "actions": TermHistory(RoboCfg.frame_stack, RoboCfg.num_actions),
    }

    def reset(self):
        for key, value in self.obs_buffer.items():
            value.reset()
            self.is_first_frame = True


class Command:
    vx, vy, dyaw = 0.0, 0.0, 0.0
    vx_increment = 0.1
    vy_increment = 0.1
    dyaw_increment = 0.1

    min_vx = -1.0
    max_vx = 2.0
    min_vy = -1.0
    max_vy = 1.0
    min_dyaw = -1.0
    max_dyaw = 1.0
    reset_requested = False

    def __init__(self):
        self.reset()
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def reset(self):
        """reset all velocities to zero"""
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0
        self.reset_requested = True
        self.terminate = False
        print(f"Velocities reseted. ")

    def get_command(self):
        """get current command"""
        vx, vy, dyaw = self.vx, self.vy, self.dyaw
        is_reset = self.reset_requested
        terminate = self.terminate
        self.reset_requested = False
        return [vx, vy, dyaw], is_reset, terminate

    def on_press(self, key):
        """Key press event handler"""
        try:
            # Number key controls: 8/5 control forward/backward (vx), 4/6 control left/right (vy), 7/9 control left/right turn (dyaw)
            if hasattr(key, 'char') and key.char is not None:
                c = key.char.lower()
                if c == '8':
                    self.vx = np.clip(self.vx + Command.vx_increment, self.min_vx, self.max_vx)
                elif c == '2':
                    self.vx = np.clip(self.vx - Command.vx_increment, self.min_vx, self.max_vx)
                elif c == '4':
                    self.vy = np.clip(self.vy + Command.vy_increment, self.min_vy, self.max_vy)
                elif c == '6':
                    self.vy = np.clip(self.vy - Command.vy_increment, self.min_vy, self.max_vy)
                elif c == '7':
                    self.dyaw = np.clip(self.dyaw + Command.dyaw_increment, self.min_dyaw, self.max_dyaw)
                elif c == '9':
                    self.dyaw = np.clip(self.dyaw - Command.dyaw_increment, self.min_dyaw, self.max_dyaw)
                elif c == '5':
                    self.terminate = True
                elif c == '0':
                    self.reset()
            print(f"cmd: {self.vx:.2f}, {self.vy:.2f}, {self.dyaw:.2f}")
        except AttributeError:
            pass


def set_joint_angle(joint_name, angle):
    joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    d.qpos[m.jnt_qposadr[joint_id]] = angle
    
def set_joint_vel(joint_name, angle):
    joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    d.qvel[m.jnt_dofadr[joint_id]] = angle

def reset(observation: History):
    mujoco.mj_resetData(m, d)
    observation.reset()

def control(target_pos, pos, target_vel, vel):
    tau = (pos - target_pos) * RoboCfg.stiffness + (vel - target_vel) * RoboCfg.damping
    return np.clip(tau, -RoboCfg.tau_limit, RoboCfg.tau_limit)

def quat_apply_inverse(quat, v):
    q_w = quat[0] 
    q_vec = quat[1:] 
    v_vec = np.array(v, dtype=np.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result


def get_sensor_data(sensor_id):
    if sensor_id == -1:
        raise ValueError(f"Sensor id '({sensor_id})' invalid!")
    start_idx = m.sensor_adr[sensor_id]
    dim = m.sensor_dim[sensor_id]
    sensor_values = d.sensordata[start_idx : start_idx + dim]
    return np.array(
        sensor_values, 
        dtype=np.float32
    )


def get_obs(actions, default_dof_pos, commands=[0.5, 0.0, 0.0]):
    base_quat = get_sensor_data(RoboCfg.imu_quat_id) # np
    gravity = [0.0, 0.0, -1.0]
    projected_gravity = quat_apply_inverse(base_quat, gravity) # np
    imu_gyro = get_sensor_data(RoboCfg.imu_gyro_id) # np

    dof_pos = np.zeros(RoboCfg.num_actions, dtype=np.float32)
    for i, id in enumerate(RoboCfg.pos_sensor_ids):
        dof_pos[i] = get_sensor_data(id)[0]

    dof_vel = np.zeros(RoboCfg.num_actions, dtype=np.float32)
    for i, id in enumerate(RoboCfg.vel_sensor_ids):
        dof_vel[i] = get_sensor_data(id)[0]

    dof_frc = np.zeros(RoboCfg.num_actions, dtype=np.float32)
    for i, id in enumerate(RoboCfg.frc_sensor_ids):
        dof_frc[i] = get_sensor_data(id)[0]

    cmds = np.array(commands, dtype=np.float32)

    return (
            cmds,  # 3
            imu_gyro * 0.5,  # 3
            projected_gravity * 1.0,  # 3
            (dof_pos - default_dof_pos) * 1.0,  # 12
            dof_vel * 0.1,  # 16
            dof_frc * 0.1, # 16
            actions.copy().astype(np.float32),  # 16
    )



def main(args):
    # 加载模型
    try:
        loaded_policy = torch.jit.load(args.load_policy, map_location=torch.device('cpu'))
        loaded_policy.eval()  # 设置为评估模式

        loaded_encoder = torch.jit.load(args.load_encoder, map_location=torch.device('cpu'))
        loaded_encoder.eval()  
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

    # 初始化观察数据
    actions = np.zeros(16, dtype=np.float32)
    ctlData = np.zeros(16, dtype=np.float32)
    target_pos = np.zeros(16, dtype=np.float32)
    target_vel = np.zeros(16, dtype=np.float32)

    cmd_listener = Command()
    observation = History()
    observation.is_first_frame = True
    def sim_step(observation, actions, ctlData, cmd_listener):
        """Run one simulation step: observe → policy → step physics.
        Returns (should_continue).
        """
        cmd, is_reset, terminate = cmd_listener.get_command()
        if is_reset:
            reset(observation)
        if terminate:
            return False

        cmds, ang_vel, pro_grav, dof_pos, dof_vel, dof_frc, last_actions = get_obs(actions, default_dof_pos=RoboCfg.default_pos, commands=cmd)

        if observation.is_first_frame:
            for key, vec in zip(OBS_HISTORY_KEYS, [cmds, ang_vel, pro_grav, dof_pos[:12], dof_vel, dof_frc, last_actions]):
                observation.obs_buffer[key].fill_tile(vec)
            observation.is_first_frame = False
        else:
            for key, vec in zip(OBS_HISTORY_KEYS, [cmds, ang_vel, pro_grav, dof_pos[:12], dof_vel, dof_frc, last_actions]):
                observation.obs_buffer[key].append(vec)

        obs_hist = np.concatenate(
            [observation.obs_buffer[key].flat() for key in OBS_HISTORY_KEYS], axis=0
        )[None, :].astype(np.float32)
        assert obs_hist.shape[1] == RoboCfg.num_observations, (
            f"Expected policy input dim {RoboCfg.num_observations}, "
            f"got {obs_hist.shape[1]}."
        )
        obs_hist = torch.tensor(obs_hist)

        with torch.inference_mode():
            encoder_out = loaded_encoder(obs_hist)
            policy_input = torch.cat([encoder_out, obs_hist], dim=-1)
            actions[:] = loaded_policy(policy_input)[0].detach().numpy()

        act = actions * RoboCfg.action_scale + RoboCfg.default_pos
        act = np.clip(act, -RoboCfg.action_clip, RoboCfg.action_clip)

        target_pos[:12] = act[:12]
        target_vel[-4:] = act[-4:]

        ctlData[RoboCfg.usd2urdf] = control(target_pos, dof_pos, target_vel, dof_vel)
        for i in range(RoboCfg.num_actions):
            d.ctrl[i] = ctlData[i]

        for i in range(DECIMATION):
            mujoco.mj_step(m, d)

        return True

    if args.video:
        import cv2

        width, height = 1920, 1080
        renderer = mujoco.Renderer(m, height, width)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = args.fps
        video_writer = cv2.VideoWriter(args.video_path, fourcc, fps, (width, height))
        mujoco.mjv_defaultFreeCamera(m, renderer.scene.camera)

        print(f"Recording video to {args.video_path}  (fps={fps})  — press Esc to stop")
        while True:
            should_continue = sim_step(
                observation, actions, ctlData, cmd_listener
            )
            if not should_continue:
                break
            renderer.update_scene(d)
            frame = renderer.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()
        renderer.close()
        print(f"Video saved to {args.video_path}")
    else:
        step_time = 1./args.fps
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running():
                step_start = time.time()
                should_continue = sim_step(
                    observation, actions, ctlData, cmd_listener
                )
                if not should_continue:
                    break
                viewer.sync()
                time_until_next_step = step_time - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPMini sim2sim deployment.")
    parser.add_argument("--load_encoder", type=str, default="encoder.pt",
                        help="Path to the JIT-compiled encoder checkpoint (.pt).")
    parser.add_argument("--load_policy", type=str, default="policy.pt",
                        help="Path to the JIT-compiled policy checkpoint (.pt).")
    parser.add_argument("--video", action="store_true",
                        help="Save rendering to video file.")
    parser.add_argument("--video_path", type=str, default="output.mp4",
                        help="Output video file path (used with --video).")
    parser.add_argument("--fps", type=int, default=60,
                        help="Video fps.")
    args = parser.parse_args()

    main(args)