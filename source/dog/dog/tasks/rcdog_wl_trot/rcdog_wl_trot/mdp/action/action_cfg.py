from __future__ import annotations

import torch
from isaaclab.envs import mdp  # noqa: F401, F403
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

from . import action


@configclass
class PosActionCfg( mdp.JointPositionActionCfg ):

    class_type: type[ActionTerm] = action.PosAction
    noise: NoiseCfg | None = None


@configclass
class VelActionCfg( mdp.JointVelocityActionCfg ):

    class_type: type[ActionTerm] = action.VelAction
    noise: NoiseCfg | None = None


