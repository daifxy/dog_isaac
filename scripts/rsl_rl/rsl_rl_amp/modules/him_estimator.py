# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..networks import MLP


class HimEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        num_one_step_priveleged_obs,
        enc_hidden_dims=[256, 128, 64],
        proj_hidden_dims=[256, 256],  # Projector hidden dims for Barlow Twins
        command_dim=3,  # default to command linear velocity and angular velocity (x, y, yaw)
        estimate_dim=3,  # default to estimate linear velocity (x, y, z)
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        barlow_lambda=5e-3,  # Barlow Twins: weight for off-diagonal terms
        projector_output_dim=256,  # Projector output dimension (higher dim for redundancy reduction)
        **kwargs
    ):
        if kwargs:
            print(
                "Estimator_CL.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        assert command_dim >= 0 and estimate_dim >= 0

        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_priveleged_obs = num_one_step_priveleged_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.barlow_lambda = barlow_lambda
        self.command_dim = command_dim
        self.estimate_dim = estimate_dim
        self.projector_output_dim = projector_output_dim

        # Shared Encoder (Backbone) - processes temporal observations
        enc_input_dim = temporal_steps * self.num_one_step_obs
        self.encoder = MLP(enc_input_dim, enc_hidden_dims[-1], enc_hidden_dims[:-1], activation)
        # Prediction layer for velocity estimation (auxiliary task)
        self.pred_layer = MLP(enc_hidden_dims[-1], estimate_dim, [128, 64], activation)

        # Projector (MLP head) - projects representations to high-dimensional space
        # This is the key component in Barlow Twins for redundancy reduction
        sizes = [enc_hidden_dims[-1]] + proj_hidden_dims + [projector_output_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(projector_output_dim, affine=False)

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs_history):
        """
        Forward pass for inference.
        Returns the state estimate and encoder representation (NOT projector output).

        Note: In Barlow Twins, the projector is only used during training for
        computing the cross-correlation matrix. For inference/policy, we use
        the encoder's representation directly.
        """
        repr = self.encoder(obs_history.detach())
        estimate = self.pred_layer(repr)
        # Return encoder representation, NOT projector output
        return estimate.detach(), repr.detach()

    def update(self, obs, critic_obs, next_obs, lr=None):
        """
        Update using Barlow Twins self-supervised learning.

        Args:
            obs: history observations at t-5:t (first view)
            critic_obs: critic observations containing ground truth state
            next_obs: history observations at t-4:t+1 (second view, shifted by 1 step)
            lr: optional learning rate
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # Extract ground truth state for estimation loss
        state = critic_obs[
            :,
            -self.num_one_step_priveleged_obs
            + self.num_one_step_obs : -self.num_one_step_priveleged_obs
            + self.num_one_step_obs
            + self.estimate_dim,
        ].detach()

        # Pass both views through the SAME encoder (key point of Barlow Twins)
        # IMPORTANT: Do NOT detach inputs - we need gradients to flow back to encoder!
        repr_1 = self.encoder(obs)
        repr_2 = self.encoder(next_obs)

        # Split into state estimate and representations
        state_estimate = self.pred_layer(repr_1)

        # Project representations to high-dimensional space (Barlow Twins projector)
        z_1 = self.projector(repr_1)
        z_2 = self.projector(repr_2)

        # Compute Barlow Twins loss
        # Cross-correlation matrix C: element C[i,j] is correlation between feature i and j
        batch_size = z_1.size(0)
        # empirical cross-correlation matrix
        c = self.bn(z_1).T @ self.bn(z_2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        # Total Barlow Twins loss
        barlow_loss = on_diag + self.barlow_lambda * off_diag

        # State estimation loss (auxiliary task)
        if self.estimate_dim == 0:
            estimation_loss = torch.tensor(0.0).to(barlow_loss.device)
        else:
            estimation_loss = F.mse_loss(state, state_estimate)

        # Combined loss
        losses = estimation_loss + barlow_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), barlow_loss.item()

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
