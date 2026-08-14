from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.envs import ManagerBasedRLEnv
import torch

from ..mdp.rewards import joint_powers_l1

class EnergyRegularizationRewardManager(RewardManager):
    negative_terms = ["", ]
    sigma_aux = 0.2        

    def compute(self, dt: float) -> torch.Tensor:
        # reset computation
        self._reward_buf[:] = 0.0
        negative_reward = torch.zeros_like(self._reward_buf)
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            if name in self.negative_terms:
                negative_reward += value
            else:
                self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt

        return self._reward_buf * torch.exp(negative_reward / self.sigma_aux) # negative_reward本身就<0
