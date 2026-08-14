from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # augment with 2 copies: original + left-right symmetric
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])

        # critic observation group
        # -- original
        obs_aug["privileged"][:batch_size] = obs["privileged"][:]
        # -- left-right
        obs_aug["privileged"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env.unwrapped, obs["privileged"])

        # encoder observation group
        # -- original
        obs_aug["labels"][:batch_size] = obs["labels"][:]
        # -- left-right
        obs_aug["labels"][batch_size : 2 * batch_size] = _transform_encoder_obs_left_right(env.unwrapped, obs["labels"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # augment with 2 copies: original + left-right symmetric
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.devic, dtype=actions.dtype)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug



"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # ang vel
    obs[:, :3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 9:21] = _switch_joints_left_right(obs[:, 9:21], is_wheels=False)
    # joint vel
    obs[:, 21:37] = _switch_joints_left_right(obs[:, 21:37], is_wheels=True)
    # joint torque
    obs[:, 37:53] = _switch_joints_left_right(obs[:, 37:53], is_wheels=True)
    # last actions
    obs[:, 53:69] = _switch_joints_left_right(obs[:, 53:69], is_wheels=True)

    return obs

def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 12:24] = _switch_joints_left_right(obs[:, 12:24], is_wheels=False)
    # joint vel
    obs[:, 24:40] = _switch_joints_left_right(obs[:, 24:40], is_wheels=True)
    # joint torque
    obs[:, 40:56] = _switch_joints_left_right(obs[:, 40:56], is_wheels=True)
    # last actions
    obs[:, 56:72] = _switch_joints_left_right(obs[:, 56:72], is_wheels=True)
    # robot_joint_acc
    obs[:, 72:88] = _switch_joints_left_right(obs[:, 72:88], is_wheels=True)
    # base_acc_vel
    obs[:, 88:91] = obs[:, 88:91] * torch.tensor([1, -1, 1], device=device)
    # feet_lin_vel
    obs[:, 91:95] = _switch_feet_left_right(obs[:, 91:95])
    # feet_contact_force
    obs[:, 95:99] = _switch_feet_left_right(obs[:, 95:99])


    return obs

def _transform_encoder_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)

    return obs

"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:], is_wheels=True)
    return actions



"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
WL_DOG_LEG_JOINT_NAMES = [
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

WL_DOG_WHEEL_JOINT_NAMES = [
    "LF_FOOT_JOINT",    
    "RF_FOOT_JOINT",  
    "LB_FOOT_JOINT",  
    "RB_FOOT_JOINT",  
]
]

"""


def _switch_feet_left_right(feet_data: torch.Tensor) -> torch.Tensor:

    feet_data_switched = torch.zeros_like(feet_data)
    # left <-- right
    feet_data_switched[..., [0, 2]] = feet_data[..., [1, 3]]
    # right <-- left
    feet_data_switched[..., [1, 3]] = feet_data[..., [0, 2]]

    return feet_data_switched




def _switch_joints_left_right(joint_data: torch.Tensor, is_wheels: bool) -> torch.Tensor:

    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [0, 1, 2, 6, 7, 8]] = joint_data[..., [3, 4, 5, 9, 10, 11]]
    # right <-- left
    joint_data_switched[..., [3, 4, 5, 9, 10, 11]] = joint_data[..., [0, 1, 2, 6, 7, 8]]

    if is_wheels:
        joint_data_switched[..., [12, 14]] = joint_data[..., [13, 15]]
        joint_data_switched[..., [13, 15]] = joint_data[..., [12, 14]]

    # Flip the sign of the HAA joints
    joint_data_switched[..., [0, 3, 6, 9]] *= -1.0

    return joint_data_switched

