# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from ..modules import MLP, EmpiricalNormalization, HiddenState
from ..modules.distribution import Distribution
from ..utils import resolve_callable, unpad_trajectories

class EncoderMLPModel(nn.Module):
    """MLP-based neural model.    """

    is_mlp_encoder = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int = 3,
        is_mlp_encoder: bool = True,
        hidden_dims: tuple[int, ...] | list[int] = [256, 256],
        activation: str = "elu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.is_mlp_encoder = is_mlp_encoder
        if self.is_mlp_encoder:
            self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
            if kwargs:
                print(
                    "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                    + str([key for key in kwargs.keys()])
                )
            # MLP
            self.output_dim = output_dim
            if output_dim <= 0:
                raise ValueError("EncoderMLPModel.__init__: output_dim must be > 0")
            self.mlp = MLP(self.obs_dim, output_dim, hidden_dims, activation)
        else:
            self.output_dim = None
            self.mlp = torch.nn.Sequential()


    def forward(self, obs) -> torch.Tensor:
        if self.is_mlp_encoder:
            # Get MLP input latent
            latent = self.get_latent(obs)
            # MLP forward pass
            return self.mlp(latent)
        else:
            return None

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        # Select and concatenate observations
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        return latent


    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def inference(self, input):
        with torch.no_grad():
            return self.mlp(input)



class _TorchMLPModel(nn.Module):
    """Exportable MLP model for JIT."""

    def __init__(self, model: EncoderMLPModel) -> None:
        """Create a TorchScript-friendly copy of an MLPModel."""
        super().__init__()
        self.mlp = copy.deepcopy(model.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        return self.mlp(x)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxMLPModel(nn.Module):
    """Exportable MLP model for ONNX."""

    def __init__(self, model: EncoderMLPModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__()
        self.verbose = verbose
        self.mlp = copy.deepcopy(model.mlp)
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        return self.mlp(x)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["mlp_input"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["mlp_output"]
