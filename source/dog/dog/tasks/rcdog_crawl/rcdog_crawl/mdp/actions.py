from __future__ import annotations

import torch
from isaaclab.envs.mdp import *

class JointPosAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: JointPosCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointPosCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clip(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset - self._asset.data.joint_pos[:, self._joint_ids]

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class JointVelAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: JointVelCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointVelCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clip(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset - self._asset.data.joint_vel[:, self._joint_ids]

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class JointPosCfg(JointActionCfg):
    class_type: type[ActionTerm] = JointPosAction


@configclass
class JointVelCfg(JointActionCfg):
    class_type: type[ActionTerm] = JointVelAction

