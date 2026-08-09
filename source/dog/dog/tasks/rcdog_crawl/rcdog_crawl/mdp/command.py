from __future__ import annotations

from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *

import torch


class Command(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.target_height = torch.full((self.num_envs,), self.cfg.target_height[1])
        self.num_crawl = self.cfg.crawl_probability * self.num_envs
        self.target_height[-int(self.num_crawl):] = self.cfg.target_height[0]
        self.target_height = self.target_height.to(self.device)

        self.commands = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.commands[:, -1] = self.target_height.clone()

    @property
    def command(self) -> torch.Tensor:
        return self.commands

    def _resample_command(self, env_ids: Sequence[int]):
        crawl_env_ids = self.target_height[env_ids] < self.cfg.target_height[1]
        prop = torch.ones(len(env_ids), device=self.device)
        prop[crawl_env_ids] *= 0.5
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x) * prop
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y) * prop
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z) * prop
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        self.commands[env_ids, :3] = self.vel_command_b[env_ids].clone()

    # def _update_command(self):
    #     super()._update_command()

@configclass
class CommandCfg(UniformVelocityCommandCfg):
    class_type: type = Command

    target_height: tuple[float, float] = MISSING
    crawl_probability: float = 0.0
