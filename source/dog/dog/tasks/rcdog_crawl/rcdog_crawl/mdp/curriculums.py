from __future__ import annotations

from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *

import numpy as np
import torch
import copy


class CommandCurriculum(ManagerTermBase):

    ranges: UniformVelocityCommandCfg.Ranges
    first_check: bool = True

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, reward_name: str, command_name: str) -> float:
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if self.first_check:
            self.ranges = copy.deepcopy(env.command_manager.get_term(command_name).cfg.ranges)
            env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x = [n*0.2 for n in self.ranges.lin_vel_x]
            env.command_manager.get_term(command_name).cfg.ranges.lin_vel_y = [n*0.2 for n in self.ranges.lin_vel_y]
            self.first_check = False
        if (torch.mean(env.reward_manager._episode_sums[reward_name][env_ids]) / env.max_episode_length) > (0.8 * env.reward_manager.get_term_cfg(reward_name).weight):
            ranges = env.command_manager.get_term(command_name).cfg.ranges
            ranges.lin_vel_x[0] = np.clip(ranges.lin_vel_x[0] - 0.2, self.ranges.lin_vel_x[0], 0.)
            ranges.lin_vel_x[1] = np.clip(ranges.lin_vel_x[1] + 0.2, 0., self.ranges.lin_vel_x[1])
            ranges.lin_vel_y[0] = np.clip(ranges.lin_vel_y[0] - 0.2, self.ranges.lin_vel_y[0], 0.)
            ranges.lin_vel_y[1] = np.clip(ranges.lin_vel_y[1] + 0.2, 0., self.ranges.lin_vel_y[1])

        return env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x[1]

