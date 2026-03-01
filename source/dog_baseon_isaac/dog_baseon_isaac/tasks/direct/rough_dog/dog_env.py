from __future__ import annotations

import math
import os
import sys
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
# from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor, RayCaster, Imu, Camera
from isaaclab.terrains import TerrainImporter
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

from .dog_env_cfg import RoughEnvCfg
import dog_baseon_isaac
utils_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(dog_baseon_isaac.__file__))))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from my_utils.debug import _MyPrint_, Visualization

def random_in_range(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size = shape, device=device) + lower

class RoughEnv(DirectRLEnv):
    cfg: RoughEnvCfg

    def __init__(self, cfg: RoughEnvCfg, render_mode: str | None = None, **kwargs):
        self.MyPrint = _MyPrint_()
        super().__init__(cfg, render_mode, **kwargs)
        self.train_mode: bool = True
        self.visualize_marks = Visualization(self.num_envs, self.device)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # -- Initialize command ranges
        self.command_ranges = torch.zeros((self.cfg.command_space, 2), dtype=torch.float32, device=self.device)
        self.command_ranges[0,0] = self.cfg.command_cfg["lin_vel_x_range"][0] * self.cfg.curriculum_cfg["curriculum_min_commands_proportion"]
        self.command_ranges[0,1] = self.cfg.command_cfg["lin_vel_x_range"][1] * self.cfg.curriculum_cfg["curriculum_min_commands_proportion"]
        self.command_ranges[1,0] = self.cfg.command_cfg["lin_vel_y_range"][0] * self.cfg.curriculum_cfg["curriculum_min_commands_proportion"]
        self.command_ranges[1,1] = self.cfg.command_cfg["lin_vel_y_range"][1] * self.cfg.curriculum_cfg["curriculum_min_commands_proportion"]
        self.command_ranges[2,0] = self.cfg.command_cfg["ang_vel_range"][0]# * self.cfg.curriculum_cfg["curriculum_min_commands_proportion"]
        self.command_ranges[2,1] = self.cfg.command_cfg["ang_vel_range"][1]# * self.cfg.curriculum_cfg["curriculum_min_commands_proportion"]

        # -- set joint position limits
        self.joint_limits = torch.zeros((self.num_envs, len(self.cfg.joints_names), 2), device=self.device, dtype=torch.float32)
        _, joints_names = self.robot.find_joints(self.cfg.joints_names, preserve_order=False)

        for name in self.cfg.joint_pos_limits.keys():
            if name not in joints_names:
                self.MyPrint(f"Joint ({name}) not found in robot.", "ERROR", True)

        joint_limits = torch.zeros((len(self.cfg.joints_names), 2), device=self.device, dtype=torch.float32)
        i = 0
        for name in joints_names:
            if name in self.cfg.joint_pos_limits:
                joint_limits[i] = torch.tensor(self.cfg.joint_pos_limits[name])
            else:
                joint_limits[i] = torch.tensor([-1e7, 1e7])
                self.MyPrint(f"The value for Joint ({name}) limit was not specified and will be set to [-1e7, 1e7].", "WARNING")
            i += 1
                
        self.joint_limits[:, :, 0] = joint_limits[:,0]
        self.joint_limits[:, :, 1] = joint_limits[:,1]
        self.robot.data.default_joint_pos_limits = self.joint_limits
        self.robot.write_joint_position_limit_to_sim(self.joint_limits)

        # -- Domain Randomization
        self.domain_rand_cfg = self.cfg.other_randomize_cfg

        self.randomize_respawn_pose_ranges      = self.domain_rand_cfg["randomize_respawn_point"]["pose_range"]
        self.randomize_respawn_velocity_range   = self.domain_rand_cfg["randomize_respawn_point"]["velocity_range"]
        
        external_force_and_torque               = self.domain_rand_cfg["external_force_and_torque"]
        self.external_forces_body               = external_force_and_torque["applied_bodies"]
        self.external_max_force                 = torch.tensor(external_force_and_torque["max_force"], device=self.device)
        self.external_max_torque                = torch.tensor(external_force_and_torque["max_torque"], device=self.device)
        if self.train_mode:
            self.applied_external_max_force     = torch.tensor(external_force_and_torque["max_force"] * external_force_and_torque["curriculum_min_force_proportion"], device=self.device)
        else:
            self.applied_external_max_force     = torch.tensor(external_force_and_torque["max_force"], device=self.device)
        self.applied_external_max_torque        = torch.tensor(external_force_and_torque["max_torque"] * external_force_and_torque["curriculum_min_force_proportion"], device=self.device)
        self.curriculum_min_force_proportion    = torch.tensor(external_force_and_torque["curriculum_min_force_proportion"], device=self.device)
        self.curriculum_force_level_growth_rate = torch.tensor(external_force_and_torque["curriculum_force_level_growth_rate"], device=self.device)
        self.curriculum_force_check_interval    = external_force_and_torque["curriculum_force_check_interval"]
        self.duration                           = external_force_and_torque["duration"]
        self.interval                           = external_force_and_torque["interval"]
        self.is_global                          = external_force_and_torque["is_global"]
        self.position                           = torch.tensor(external_force_and_torque["position"], device=self.device)

        # randomize
        randomize_base_com_range: dict          = self.domain_rand_cfg["randomize_base_com_range"]
        randomize_other_com_range: dict         = self.domain_rand_cfg["randomize_other_com_range"]
        randomize_physics_scene_gravity         = self.domain_rand_cfg["randomize_physics_scene_gravity"]
        randomize_rigid_body_collider_offsets   = self.domain_rand_cfg["randomize_rigid_body_collider_offsets"]

        mdp.randomize_rigid_body_com(
            self,
            self.scene._ALL_INDICES, 
            randomize_base_com_range,
            SceneEntityCfg("Robot", body_names="base_link"),
        )
        mdp.randomize_rigid_body_com(
            self,
            self.scene._ALL_INDICES, 
            randomize_other_com_range,
            SceneEntityCfg("Robot", body_names=".*_Link"),
        )
        mdp.randomize_physics_scene_gravity(
            self,
            self.scene._ALL_INDICES,
            randomize_physics_scene_gravity["gravity_distribution_params"],
            randomize_physics_scene_gravity["operation"],
            randomize_physics_scene_gravity["distribution"]
        )
        # mdp.randomize_rigid_body_collider_offsets(
        #     self,
        #     self.scene._ALL_INDICES,
        #     SceneEntityCfg("Robot", body_names=".*"),
        #     randomize_rigid_body_collider_offsets["rest_offset_distribution_params"],
        #     randomize_rigid_body_collider_offsets["contact_offset_distribution_params"],
        #     randomize_rigid_body_collider_offsets["distribution"],
        # )

        # -- Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies(self.cfg.die_if_contact_bodies)
        self._feet_ids, _ = self._contact_sensor.find_bodies(self.cfg.feet_contact_bodies)
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.undesired_contact_bodies)
        self._external_forces_body_ids, _ = self.robot.find_bodies(self.external_forces_body ,preserve_order=True)

        # -- Initialize buffers
        self.commands = torch.zeros((self.num_envs, self.cfg.command_space), device=self.device, dtype=torch.float32)
        self.enable_external_force = self.domain_rand_cfg["external_force_and_torque"]["enabled"]
        self.num_bodies = len(self._external_forces_body_ids)
        self.external_forces_duration       = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.external_forces_free_interval  = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._forces     = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
        self._torques    = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
        self._positions  = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
        self._ud_t: int = 0
        self.direction = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
        self.random_interval = torch.normal(self.interval, 0.75, (self.num_envs,), device=self.device).abs() + 0.1

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self._previous_actions  = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self._processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self.previous_ang_vel = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        # self.ang_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.last_processed_actions = self._processed_actions.clone()

        self.fall = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.height_data = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        if self.cfg.state_space != self.cfg.observation_space:
            self.privileged_obs_buf = torch.zeros((self.num_envs, self.cfg.state_space), device=self.device, dtype=torch.float32)
        self.obs_buf = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device, dtype=torch.float32)
        self.history_obs = torch.zeros((self.num_envs, self.cfg.env_cfg["history_length"], self.cfg.env_cfg["policy_slice_obs"]), device=self.device, dtype=torch.float32)
        self.slice_obs_buf = torch.zeros((self.num_envs, self.cfg.env_cfg["policy_slice_obs"]), device=self.device, dtype=torch.float32)
        
        joints = self.cfg.joints_names
        self.apply_action_ids, names = self.robot.find_joints(joints, preserve_order=True)
        self.curriculum_steps: int = 0
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.commands_level: int = 0
        self.obs_scales = self.cfg.obs_scales
        self.noise_scale_vec = self._get_noise_scale_vec()
        
        # -- prepare reward functions and multiply reward scales by dt
        self.episode_sums = dict()
        self.reward_scales: dict = self.cfg.reward_cfg["reward_scales"]
        self.reward_functions = dict()
        for name in self.reward_scales.keys():
            if name == "termination":
                self.episode_sums["termination"] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
                continue
            self.reward_scales[name] *= self.scene.physics_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # -- terrain curriculum
        if self.enable_terrain:
            # self._terrain.terrain_levels = torch.clip(self._terrain.terrain_levels - 50, min=0)
            # self._terrain.env_origins = self._terrain.terrain_origins[self._terrain.terrain_levels, self._terrain.terrain_types]
            self.robot.data.default_root_state[:, :3] = self._terrain.env_origins
            self.robot.data.default_root_state[:, 2] += 0.365
            

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add articulation to scene
        self.scene.articulations["Robot"] = self.robot
        # terrain
        self.enable_terrain = self.cfg.enable_terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)
        # sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._imu = Imu(self.cfg.imu_sensor)
        self.scene.sensors["imu"] = self._imu
        # self._camera = Camera(self.cfg.camera_sensor)
        # self.scene.sensors["camera"] = self._camera
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.MyPrint("CPU simulation is not supported!!!", "ERROR", True)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def set_commands(self, envs_idx, commands):
        self.commands[envs_idx] = torch.tensor(commands, device=self.device, dtype=torch.float32)

    def set_terrain_id(self, terrain_id, terrain_level):
        if terrain_id is not None:
            self._terrain.terrain_types[:] = terrain_id
            self._terrain.terrain_types = torch.clip(self._terrain.terrain_types, min=0, max=self._terrain.terrain_origins.shape[1] -1)
            self.MyPrint(f"Set terrain type to {self._terrain.terrain_types.item()}.", "DEBUG")
        if terrain_level is not None:
            self._terrain.terrain_levels[:] += terrain_level
            self._terrain.terrain_levels = torch.clip(self._terrain.terrain_levels, min=0, max=self._terrain.max_terrain_level-1)
            self.MyPrint(f"Set terrain level to {self._terrain.terrain_levels.item()}.", "DEBUG")
        self.robot.data.default_root_state[:, :3] = self._terrain.terrain_origins[self._terrain.terrain_levels, self._terrain.terrain_types]
        self.robot.data.default_root_state[:, 2] += 0.365
        self._reset_idx(None)

    def get_camera_data(self):
        return self._camera.data.output["rgb"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions)
        self._processed_actions = (self._actions * self.cfg.action_scale) + self.robot.data.default_joint_pos - self.robot.data.joint_pos
        self._processed_actions = torch.clip(self._processed_actions, self.joint_limits[:,:,0], self.joint_limits[:,:,1])
        self._processed_actions = self._processed_actions*0.7 + self.last_processed_actions*0.3
        # if self.cfg.noise_cfg["dead_zone"] > 0.0:
        #     action_ch = self._processed_actions - self.last_processed_actions
        #     less_than = action_ch.abs() < self.cfg.noise_cfg["dead_zone"]
        #     self._processed_actions[less_than] = self.last_processed_actions[less_than]
        self.last_processed_actions = self._processed_actions.clone()
    
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions, self.apply_action_ids)
        # 基础命令训练好后开始加入外部力干扰
        # if self.enable_external_force or not self.train_mode:
        #     self.set_external_force_and_torque()

    def _get_observations(self,) -> dict:
        # update previous actions
        self._previous_actions = self._actions.clone()
        # self.ang_acc = (self.robot.data.root_com_ang_vel_b - 2*self.previous_ang_vel[:,0] +  self.previous_ang_vel[:,1])/(self.physics_dt*2)
        # self.previous_ang_vel[:,1] = self.previous_ang_vel[:,0].clone()
        # self.previous_ang_vel[:,0] = self.robot.data.root_com_ang_vel_b.clone()
        # resample commands
        resample_ids = (self.episode_length_buf % self.cfg.resample_commands_length == 0).nonzero().flatten()
        self.resample_commands(resample_ids)
        # feet contact forces
        # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # feet_contact_forces = torch.norm(net_contact_forces[:, 0, self._feet_ids], dim=-1)>1.0

        # update observation buffer
        self.slice_obs_buf = torch.cat((    self.robot.data.root_com_lin_vel_b * self.cfg.obs_scales["lin_vel"],
                                            self.robot.data.root_com_ang_vel_b * self.cfg.obs_scales["ang_vel"],
                                            # self._imu.data.lin_acc_b * self.cfg.obs_scales["lin_acc"],
                                            self.robot.data.projected_gravity_b,
                                            self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                                            self.robot.data.joint_vel * self.cfg.obs_scales["dof_vel"],
                                            ),dim=-1,)
        if self.cfg.noise_cfg["noise"]:
            self.slice_obs_buf += (2 * torch.randn_like(self.slice_obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = torch.cat([self.history_obs, self.slice_obs_buf.unsqueeze(1)], dim=1).view(self.num_envs, -1)
        self.obs_buf = torch.cat([self.obs_buf, self._actions], dim=-1)
        self.obs_buf = torch.cat([self.obs_buf, self.commands], axis=-1)
        if self.cfg.env_cfg["history_length"] > 1:
            self.history_obs[:, :-1, :] = self.history_obs[:, 1:, :].clone() # 移位操作
        self.history_obs[:, -1, :] = self.slice_obs_buf 
        
        clip_obs = self.cfg.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        observations = {"policy": self.obs_buf}

        if self.cfg.state_space != self.cfg.observation_space:
            self.privileged_obs_buf = self._get_states()
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            observations["privileged"] = self.privileged_obs_buf

        return observations
    
    def _get_states(self):
        self.height_data[:] = 0.0
        if self.enable_terrain:
            self.height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.14
            ).clip(-1.0, 1.0).mean(dim=-1, keepdim=True)
        else:
            self.height_data = (self.robot.data.root_com_pos_w[:, 2].unsqueeze(1)-0.12).clip(-1.0, 1.0)
        self.privileged_obs_buf = torch.cat((#self.robot.data.root_com_lin_vel_b * self.cfg.obs_scales["lin_vel"],
                                         self.height_data,),dim=-1,)
        return self.privileged_obs_buf

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        if self.train_mode:
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                total_reward += rew
                self.episode_sums[name] += rew
        
            if self.cfg.reward_cfg["only_positive_rewards"]:
                total_reward = torch.clip(total_reward, min=0.0)
            if "termination" in self.reward_scales:
                rew = self._reward_termination() * self.reward_scales["termination"]
                total_reward += rew
                self.episode_sums["termination"] += rew

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train_mode:
            self._ud_t += 1
            if self.enable_external_force and (self._ud_t % (self.interval*50) == 0):
                self._ud_t = 0
                self._push_robots()
            # 计算结束条件
            self.time_out = self.episode_length_buf > self.max_episode_length
            # 身体触地过重死亡
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            self.fall = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        else:
            self.time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return self.fall, self.time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        elif len(env_ids) == 0:
            return
        
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length / 2))
        self._actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.last_processed_actions[env_ids] = 0.0
        # self.previous_ang_vel[env_ids] = 0.0
        # self.ang_acc[env_ids] = 0.0     
        self.history_obs[env_ids] = 0.0
        # Sample new commands 
        if self.train_mode:
            if self.enable_terrain:
                self._update_terrain_curriculum(env_ids)
                self.robot.data.default_root_state[env_ids, :3] = self._terrain.env_origins[env_ids]
                self.robot.data.default_root_state[env_ids, 2] += 0.365
            self.curriculum_steps += 1
            if self.curriculum_steps % self.cfg.curriculum_cfg["curriculum_commands_check_interval"] == 0:
                self.curriculum_steps = 0
                self.update_command_curriculum(env_ids)
                self.MyPrint(f"commands_level: {self.commands_level}", "RED")
            self.resample_commands(env_ids)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        if self.train_mode:
            mdp.reset_root_state_uniform(self, env_ids, self.randomize_respawn_pose_ranges, self.randomize_respawn_velocity_range, SceneEntityCfg("Robot"))
        else:
            default_root_state = self.robot.data.default_root_state[env_ids]
            self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)     
            self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)  

        # Logging
        extras = dict()
        for key in self.episode_sums.keys():
            episodic_sum_avg = torch.mean(self.episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        if self.enable_terrain:
            extras["Episode_Reward/terrain_level"] = self._terrain.terrain_levels.float().mean()
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = 1.
        self.robot.data.root_com_vel_w[:, :2] = random_in_range(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.robot.data.root_com_vel_w[:, 2:3] = random_in_range(-0.5, 0.5, (self.num_envs, 1), device=self.device) # lin vel x/y
        self.robot.write_root_velocity_to_sim(self.robot.data.root_com_vel_w, torch.arange(self.num_envs, device=self.device))

    def set_external_force_and_torque(self): 
        ''' Called in `self._apply_action()`
            Apply random external force specific body 
            These only apply when you call: `asset.write_data_to_sim()`, however, it is called in `self.step()`
        '''
        # 完成外部力的环境开始计时到下一次执行外部力的间隔时间，未完成的环境间隔时间重置为0
        force_in_progress = self.external_forces_duration >= 0
        self.external_forces_free_interval[force_in_progress] = -1
        self.external_forces_free_interval[~force_in_progress] += self.physics_dt

        # 更新随机的外部力的持续时间阈值，未执行外部力的环境的间隔时间大于阈值的将更新外部力目标持续时间,由上一步可知`if_update_force`隐含了外部力已完成的环境条件
        if self._ud_t > 1500: # 每1500个仿真步(3s)更新一次持续时间阈值,节省计算资源
            self.random_interval = torch.normal(self.interval, 0.75, (self.num_envs,), device=self.device).abs() + 0.1
            self._ud_t = 0
        if_update_force = (self.external_forces_free_interval >= self.random_interval)
        if_update_force_flatten = if_update_force.nonzero().flatten()

        # 生成随机的方向和力
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
        if len(if_update_force_flatten) > 0:
            self.external_forces_duration[if_update_force_flatten] = random_in_range(self.duration[0], self.duration[1], if_update_force_flatten.shape, self.device)
            self.direction[if_update_force_flatten] = torch.randn((len(if_update_force_flatten), self.num_bodies, 3), device=self.device, dtype=torch.float32) + 1e-6
            self.direction[if_update_force_flatten,:,2] *= 0.5 # 减少z轴方向上的力
            self.direction[if_update_force_flatten,:,:2] *= 2.
            unit_vector = self.direction[if_update_force_flatten] / torch.norm(self.direction[if_update_force_flatten], dim=-1, keepdim=True)
            forces[if_update_force_flatten] = torch.rand((len(if_update_force_flatten), self.num_bodies, 1), device=self.device, dtype=torch.float32) * self.applied_external_max_force
            positions = torch.randn((len(if_update_force_flatten), self.num_bodies, 3), device=self.device, dtype=torch.float32) * self.position

            # 更新外部力，已完成外部力的环境将力重置为0，需要更新外部力的环境（if_update_force）将力设置为随机力，暂时不考虑施加扭矩
            self._forces[if_update_force_flatten] = unit_vector * forces[if_update_force_flatten].clone()
            self._positions[if_update_force_flatten] = positions.clone()

        if force_in_progress.any():
            self._forces[~force_in_progress, ...] = 0.0
            # self.external_forces_attenuation[force_in_progress] += self.physics_dt
            # self.external_forces_attenuation[~force_in_progress] = 0.0
        
        self.external_forces_duration -= self.physics_dt
        self._ud_t += 1
        
        if not self.train_mode:
            self.draw_force_marks(self._forces)

        # 将力写入仿真
        self.robot.set_external_force_and_torque(self._forces, self._torques, self._positions, self._external_forces_body_ids, None, self.is_global)

    def draw_force_marks(self, forces:torch.Tensor):
        direction = self.direction.squeeze(1)
        forces_ = forces.squeeze(1)
        _positions = self._positions.squeeze(1)
        _positions = _positions + self.robot.data.root_pos_w
        z_zhou = torch.tensor([[0.,0.,1.0]], device=self.device, dtype=torch.float32)
        zhou = torch.cross(z_zhou, direction, dim=-1)
        zhou = zhou / (torch.norm(zhou, dim=-1, keepdim=True)+ 1e-6)
        dot = (direction * z_zhou).sum(dim=-1, keepdim=True)
        angle = torch.arccos(dot)
        zhou *= torch.sin(angle/2)
        w = torch.cos(angle/2).view(-1, 1)
        quat = torch.cat([w, zhou], dim=-1)
        scale = torch.norm(forces_, dim=-1, keepdim=True)*2
        scale = torch.where(scale>0.1, scale, torch.zeros_like(scale))
        scale = torch.cat([scale, torch.ones((self.num_envs, 2), device=self.device, dtype=torch.float32)*0.2], dim=-1)
        self.visualize_marks.visualize(_positions, quat, scale, [0])

    def curriculum_external_force_and_torque(self):
        self.curriculum_force_steps += 1
        if self.curriculum_force_steps > self.curriculum_force_check_interval:
            self.curriculum_force_steps = 0
            if self.survive_ratio > 0.9:
                self.applied_external_max_force  += self.curriculum_force_level_growth_rate * self.applied_external_max_force
                self.applied_external_max_torque += self.curriculum_force_level_growth_rate * self.applied_external_max_torque
                self.applied_external_max_force = torch.clip(self.applied_external_max_force,
                                                     min=self.curriculum_min_force_proportion*self.external_max_force,
                                                     max=self.external_max_force)
                self.applied_external_max_torque = torch.clip(self.applied_external_max_torque,
                                                     min=self.curriculum_min_force_proportion*self.external_max_torque,
                                                     max=self.external_max_torque)

    def resample_commands(self, env_ids: torch.Tensor | None = None) -> None:
        if len(env_ids) == 0:
            return
        for idx in range(self.cfg.command_space):
            self.commands[env_ids, idx] = torch.zeros_like(self.commands[env_ids, idx]).uniform_(
                self.command_ranges[idx][0], self.command_ranges[idx][1]
            )

    def update_command_curriculum(self, env_ids):
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.commands_level += 1
            self.command_ranges[0, 0] = torch.clip(self.command_ranges[0, 0] - 0.3, -self.cfg.command_cfg["lin_vel_x_range"][0], 0.)
            self.command_ranges[0, 1] = torch.clip(self.command_ranges[0, 1] + 0.3, 0., self.cfg.command_cfg["lin_vel_x_range"][1])
            self.command_ranges[1, 0] = torch.clip(self.command_ranges[1, 0] - 0.3, -self.cfg.command_cfg["lin_vel_y_range"][0], 0.)
            self.command_ranges[1, 1] = torch.clip(self.command_ranges[1, 1] + 0.3, 0., self.cfg.command_cfg["lin_vel_y_range"][1])

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        distance = torch.norm(self.robot.data.root_com_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > (15. / 3)
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.slice_obs_buf[0])
        self.add_noise = self.cfg.noise_cfg["noise"]
        noise_scales = self.cfg.noise_cfg["noise_scales"]
        noise_level = self.cfg.noise_cfg["noise_level"]
        noise_vec[:3] = noise_scales["lin_vel"] * noise_level * self.obs_scales["lin_vel"]
        noise_vec[3:6] = noise_scales["ang_vel"] * noise_level * self.obs_scales["ang_vel"]
        # noise_vec[6:9] = noise_scales["lin_acc"] * noise_level * self.obs_scales["lin_acc"]
        noise_vec[6:9] = noise_scales["gravity"] * noise_level
        noise_vec[9:21] = noise_scales["dof_pos"] * noise_level * self.obs_scales["dof_pos"]
        noise_vec[21:33] = noise_scales["dof_vel"] * noise_level * self.obs_scales["dof_vel"]
        # noise_vec[36:48] = 0. # previous actions
        return noise_vec

# ----------------------------------------------------------------------------------------------------- reward functions

    def _reward_tracking_lin_vel(self)-> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.robot.data.root_com_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / self.cfg.reward_cfg["tracking_lin_vel_sigma"])
        return lin_vel_error_mapped
        
    def _reward_tracking_ang_vel(self):
        yaw_rate_error = torch.square(self.commands[:, 2] - self.robot.data.root_com_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / self.cfg.reward_cfg["tracking_ang_sigma"])
        return yaw_rate_error_mapped

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out

    def _reward_lin_vel_z(self):
        return torch.square(self.robot.data.root_lin_vel_b[:, 2])

    def _reward_tracking_height(self):
        height_error = torch.square(self.height_data - 0.3).squeeze(dim=1)
        return height_error

    def _reward_orientation(self):
        reward = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        return reward

    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.robot.data.joint_acc), dim=1) 

    def _reward_action_rate(self):
        action_rate = self._previous_actions - self._actions
        rew = torch.sum(torch.square(action_rate), dim=1) 
        return rew 

    def _reward_survive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_base_ang_acc(self):
        return torch.sum(torch.square(self.ang_acc[:,:2]), dim=1)

    def _reward_base_ang_vel(self):
        return torch.sum(torch.square(self.robot.data.root_com_ang_vel_b[:, :2]), dim=1)

    def _reward_diagonal_thigh_calf_similar(self):
        thigh_id, jnt_names = self.robot.find_joints(["FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint"], preserve_order=True)
        thigh = self.robot.data.joint_pos[:, thigh_id] - self.robot.data.default_joint_pos[:, thigh_id]
        front = (thigh[:, 0] * thigh[:, 1]) >= 0.
        back = (thigh[:, 2] * thigh[:, 3]) >= 0.
        rew = front*1. + back*1.
        # rew =  torch.square(self.robot.data.joint_pos[:,thigh_id[0]] + self.robot.data.joint_pos[:,thigh_id[1]] - 2*self.robot.data.default_joint_pos[:, thigh_id[0]])
        # rew += torch.square(self.robot.data.joint_pos[:,thigh_id[2]] + self.robot.data.joint_pos[:,thigh_id[3]] - 2*self.robot.data.default_joint_pos[:, thigh_id[2]])
        return rew * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_same_side_hip_similar(self):
        hip_id, jnt_names = self.robot.find_joints(["FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint"], preserve_order=True)
        rew =  torch.square(self.robot.data.joint_pos[:,hip_id[0]] + self.robot.data.joint_pos[:,hip_id[1]])
        rew += torch.square(self.robot.data.joint_pos[:,hip_id[2]] + self.robot.data.joint_pos[:,hip_id[3]])
        return rew
    
    def _reward_default_hip(self):
        hip_id, _ = self.robot.find_joints(["FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint"], preserve_order=True)
        rew = torch.square(self.robot.data.joint_pos[:,hip_id] - self.robot.data.default_joint_pos[:,hip_id]).sum(dim=1)
        return rew * (torch.abs(self.commands[:, 2]) < 0.2)
            
    def _reward_feet_air_time(self):
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        contact = torch.sum(first_contact*1., dim=1)
        air_time = (torch.sum((last_air_time - 0.5) * first_contact, dim=1)) * (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )
        n_z = contact > 0.
        air_time[n_z] /= contact[n_z]
        return air_time

    def _reward_undesired_contacts(self):
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 0.1
        contacts = torch.sum(is_contact*1., dim=1)
        return contacts

    def _reward_joint_vel(self):
        rew = ((self.robot.data.joint_vel).abs() - 10.0).clip(min=0)
        return torch.sum(torch.square(rew), dim=1)
    
    def _reward_joint_force(self):
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 10.
        contacts = torch.sum(is_contact*1., dim=1)
        return contacts

    def _reward_joint_torques(self):
        return torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
    
    def _reward_feet_force(self):
        net_contact_forces = self._contact_sensor.data.net_forces_w
        # is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0]
        contacts = torch.sum(torch.norm(net_contact_forces[:, self._feet_ids], dim=-1), dim=-1)
        rew = contacts > 50.
        return rew*1

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.robot.data.joint_pos - self.robot.data.default_joint_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.robot.data.joint_pos - self.robot.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.)
        out_of_limits += (self.robot.data.joint_pos - self.robot.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_airborne_penalty(self):
        # Penalize when feet are all airborne
        net_contact_forces = self._contact_sensor.data.net_forces_w
        contacts = torch.norm(net_contact_forces[:, self._feet_ids], dim=-1) < 0.1
        rew = torch.all(contacts, dim=1)
        return rew*1
