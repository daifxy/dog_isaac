from __future__ import annotations

from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *

import torch


def modify_env_origin(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> CurriculumTermCfg:
    """ Modify the environment origin. """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device) 
    asset: Articulation = env.scene["robot"]
    asset.data.default_root_state[env_ids, :3] = env.scene.terrain.env_origins[env_ids]
    asset.data.default_root_state[env_ids, 2] += 0.365

