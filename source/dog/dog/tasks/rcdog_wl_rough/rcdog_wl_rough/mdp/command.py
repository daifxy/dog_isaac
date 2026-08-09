from __future__ import annotations

from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *

import torch


class Command(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # self.target_height = torch.full((self.num_envs,), self.cfg.target_height[1])
        self.num_climb = int(self.cfg.probability * self.num_envs)
        # self.target_height[-int(self.num_crawl):] = self.cfg.target_height[0]
        # self.target_height = self.target_height.to(self.device)

        self.commands = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # self.commands[:, -1] = self.target_height.clone()

    @property
    def command(self) -> torch.Tensor:
        return self.commands

    def _resample_command(self, env_ids: Sequence[int]):
        # crawl_env_ids = self.target_height[env_ids] < self.cfg.target_height[1]
        # prop = torch.ones(len(env_ids), device=self.device)
        # prop[crawl_env_ids] *= 0.5
        # sample velocity commands
        length = torch.norm(torch.tensor([self.cfg.ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_y[1]], device=self.device), dtype=torch.float32)
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y) #* (1 - torch.abs(self.vel_command_b[env_ids, 0])/self.cfg.ranges.lin_vel_x[1]).clip(min=0.5, max=1.0)
        # -- ang vel yaw - rotation around z
        b = (1 - torch.norm(self.vel_command_b[env_ids, :2], dim=1)/length).clip(min=0.2, max=1.0)
        yaw_max = self.cfg.ranges.ang_vel_z[1] * b
        yaw_min = self.cfg.ranges.ang_vel_z[0] * b
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z).clip(min=yaw_min, max=yaw_max)

        self.vel_command_b[-self.num_climb:, 1:] = 0.0
        self.vel_command_b[-self.num_climb:, 0] = self.vel_command_b[-self.num_climb:, 0].clip(min=0.0, max=self.cfg.ranges.lin_vel_x[1]/1.5)
        death = self.vel_command_b < 0.1
        self.vel_command_b[death] = 0.0

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        self.commands[env_ids] = self.vel_command_b[env_ids].clone()


@configclass
class CommandCfg(UniformVelocityCommandCfg):
    class_type: type = Command

    # target_height: tuple[float, float] = MISSING
    probability: float = 0.0
