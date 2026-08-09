from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from collections.abc import Sequence
from isaaclab.assets import Articulation
import torch
from ..dog_env_cfg import DogEnvCfg
import gymnasium as gym
from dog.tasks.rough_dog.rough_dog import mdp

import copy
import numpy as np

from my_utils.debug import Visualization, MyPrint

class DogEnv(ManagerBasedRLEnv):
    """Wrapper for the WLLab environment."""

    cfg: DogEnvCfg

    train_mode: bool
    def __init__(self, cfg: DogEnvCfg, render_mode: str | None = None, **kwargs):
        self.cycle_length = cfg.cycle_length
        super().__init__(cfg, render_mode, **kwargs)
        self.visualizer = Visualization(self.num_envs, self.device)
        self.print = MyPrint()
        self.train_mode = True
        self.extras["terminal_observations"] = {"env_ids": 0, "policy": 0}

    def step(self, action: torch.Tensor):
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        
        if self.train_mode:
            # -- check terminations
            self.reset_buf = self.termination_manager.compute()
            self.reset_terminated = self.termination_manager.terminated
            self.reset_time_outs = self.termination_manager.time_outs
            # -- reward computation
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
            if self.cfg.only_positive_rewards:
                self.reward_buf[self.reward_buf < 0] = 0.
        else:
            self.reward_buf = torch.zeros(self.num_envs, device=self.device)
            self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.reset_time_outs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)


        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        if self.train_mode:
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.extras["terminal_observations"]["env_ids"] = reset_env_ids.clone()
            self.extras["terminal_observations"]["policy"] = self.obs_buf["policy"][reset_env_ids].clone()
            if len(reset_env_ids) > 0:
                # trigger recorder terms for pre-reset calls
                self.recorder_manager.record_pre_reset(reset_env_ids)

                self._reset_idx(reset_env_ids)

                # if sensors are added to the scene, make sure we render to reflect changes in reset
                if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                    for _ in range(self.cfg.num_rerenders_on_reset):
                        self.sim.render()

                # trigger recorder terms for post-reset calls
                self.recorder_manager.record_post_reset(reset_env_ids)

            # -- update command
            self.command_manager.compute(dt=self.step_dt)
            # -- step interval events
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        
        # pos = self.scene.terrain.terrain_origins.reshape(-1, 3)
        # self.visualizer.visualize(pos, 
        #                           torch.tensor([[1., 0, 0, 0]],device=self.device).repeat(60,1),
        #                           torch.tensor([[1., 1, 1]],device=self.device).repeat(60,1),
        #                           [1])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        # update the curriculum for environments that need a reset
        if self.train_mode:
            self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0


    def get_phase(self) -> torch.Tensor:
        return ((self.episode_length_buf * self.step_dt) % self.cycle_length / self.cycle_length)


    def set_commands(self, commands: torch.Tensor):
        # terr = torch.mean(self.scene["height_scanner"].data.ray_hits_w[..., 2], dim=1)
        # print(self.scene["robot"].data.root_pos_w[:, 2] - terr)
        # height = self.command_manager.get_term("base_velocity").cfg.target_height
        self.command_manager.get_term("base_velocity").command[:, :3] = torch.tensor([commands[:3],], device=self.device).repeat(self.num_envs, 1)
        # if commands[3]:
        #     self.command_manager.get_term("base_velocity").command[:, 3] = height[0]
        # else:
        #     self.command_manager.get_term("base_velocity").command[:, 3] = height[1]

    def set_terrain(self, terrain_level, terrain_type):
        if terrain_type is not None:
            self.scene.terrain.terrain_types[:] += terrain_type
            self.scene.terrain.terrain_types = torch.clip(self.scene.terrain.terrain_types, min=0, max=self.scene.terrain.terrain_origins.shape[1] -1)
            self.print(f"Set terrain type to {self.scene.terrain.terrain_types.item()}.", "DEBUG")
        if terrain_level is not None:
            self.scene.terrain.terrain_levels[:] += terrain_level
            self.scene.terrain.terrain_levels = torch.clip(self.scene.terrain.terrain_levels, min=0, max=self.scene.terrain.max_terrain_level-1)
            self.print(f"Set terrain level to {self.scene.terrain.terrain_levels.item()}.", "DEBUG")
        self.scene.terrain.env_origins = self.scene.terrain.terrain_origins[self.scene.terrain.terrain_levels, self.scene.terrain.terrain_types]
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
