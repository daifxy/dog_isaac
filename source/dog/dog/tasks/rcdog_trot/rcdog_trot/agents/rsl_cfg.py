from isaaclab.utils import configclass
from typing import Literal

from dataclasses import MISSING

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlBaseRunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

@configclass
class HimActorCriticCfg( RslRlPpoActorCriticCfg ):
    class_name: str = "HimActorCritic"
    encoder_hidden_dims: list = [256, 256]
    projector_hidden_dims: list = [256, 256]
    projector_output_dim: int = 256
    command_dim: int = 3
    estimate_dim: int = 3


@configclass
class RslRlMLPModelCfg:
    """Configuration for the MLP model."""

    class_name: str = "MLPModel"
    """The model class name. Defaults to MLPModel."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP network."""

    activation: str = MISSING
    """The activation function for the MLP network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution. Defaults to None, in which case no distribution is used."""

    stochastic: bool = MISSING
    """Whether the model output is stochastic.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and set it to None
    for deterministic output or to a valid configuration class, e.g., `GaussianDistributionCfg` for stochastic output.
    """

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the model.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `init_std` field of the distribution configuration to specify the initial noise standard deviation.
    """

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the model. Defaults to scalar.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `std_type` field of the distribution configuration to specify the type of noise standard deviation.
    """

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Defaults to False.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use
    the `HeteroscedasticGaussianDistributionCfg` if state-dependent standard deviation is desired.
    """




@configclass
class MyOnPolicyRunnerCfg( RslRlBaseRunnerCfg ):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "HimOnPolicyRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    policy: HimActorCriticCfg = None
    """The policy configuration."""

@configclass
class MyPpoAlgorithmCfg:
    class_name="HIMPPO",
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,   



@configclass
class MLPModelCfg:
    """Configuration for the MLP model."""

    class_name: str = "MLPModel"
    """The model class name. Defaults to MLPModel."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP network."""

    activation: str = MISSING
    """The activation function for the MLP network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: RslRlMLPModelCfg.DistributionCfg | None = None
    """The configuration for the output distribution. Defaults to None, in which case no distribution is used."""

@configclass
class EncoderMLPModelCfg:
    """Configuration for the encoder MLP model."""

    class_name: str = "EncoderMLPModel"
    """The model class name. Defaults to EncoderMLPModel."""
 
    is_mlp_encoder: bool = True
    """Whether the model is an MLP encoder. Defaults to True.
    
        If False, do not create the encoder network.
    """

    hidden_dims: list[int] = [256, 256]
    """The hidden dimensions of the MLP network."""

    activation: str = "elu"
    """The activation function for the MLP network."""

    output_dim: int = 3
    """The output dimension of the encoder. Defaults to 3."""


@configclass
class EncoderRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    est_learning_rate: float = 1e-4
    """The learning rate for the encoder network. Defaults to 1e-4."""