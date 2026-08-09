from __future__ import annotations

import torch
from isaaclab.envs import mdp  # noqa: F401, F403
# from isaaclab.utils import configclass
# from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg
# from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

from . import action_cfg

class PosAction( mdp.JointPositionAction ):
    cfg: action_cfg.PosActionCfg
    pass

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if self.cfg.noise is not None:
            self._processed_actions = self.cfg.noise.func(self._processed_actions, self.cfg.noise)

class VelAction( mdp.JointVelocityAction ):
    cfg: action_cfg.PosActionCfg
    pass

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if self.cfg.noise is not None:
            self._processed_actions = self.cfg.noise.func(self._processed_actions, self.cfg.noise)

