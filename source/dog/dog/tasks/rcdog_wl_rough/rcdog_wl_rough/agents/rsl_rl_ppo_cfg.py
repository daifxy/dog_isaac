# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticCfg, 
    RslRlPpoAlgorithmCfg, 
    RslRlDistillationRunnerCfg, 
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlSymmetryCfg
)
from dog.tasks.rcdog_wl_rough.rcdog_wl_rough.mdp.symmetry import wl_rough

from .rsl_cfg import (
    HimActorCriticCfg, 
    MyOnPolicyRunnerCfg, 
    MyPpoAlgorithmCfg, 
    MLPModelCfg, 
    RslRlMLPModelCfg, 
    EncoderMLPModelCfg,
    EncoderRslRlPpoAlgorithmCfg)

@configclass
class HimPPORunnerCfg(MyOnPolicyRunnerCfg):
    class_name: str = "OnPolicyRunner"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["privileged"],
    }
    num_steps_per_env = 25
    max_iterations = 20000
    save_interval = 500
    logger = "tensorboard"
    experiment_name = "rcdog_wl"
    run_name = "rcdog_wl"

    policy = HimActorCriticCfg(
        class_name="HimActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        encoder_hidden_dims=[256, 128, 64],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims=[256, 256],  # Projector hidden dims for Barlow Twins
        projector_output_dim=256,  # Projector output dim (only used in training)
        command_dim=3,  # default to command linear velocity and angular velocity (x, y, yaw)
        estimate_dim=3,  # default to estimate linear velocity (x, y, z)
        activation="elu",
    )

    algorithm = MyPpoAlgorithmCfg(
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
    )

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
        For rsl-rl >= 4.0.0, `RslRlPpoActorCriticCfg` is deprecated. Please use `RslRlMLPModelCfg` instead.
        The project's rsl_rl == 5.0.1
    """

    default_sets = ["actor", "critic", "encoder", "labels"]
    obs_groups = {
        "actor":   ["policy"],
        "critic":  ["privileged"],
        "encoder": ["policy"],
        "labels":  ["labels"], # For encoder
    }

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "wl_rough"
    run_name = "wl_rough"
    check_for_nan = True

    actor: MLPModelCfg = MLPModelCfg(
        class_name= "MLPModel",
        hidden_dims= [512, 256, 128],
        activation= "elu",
        obs_normalization= True,
        distribution_cfg= RslRlMLPModelCfg.GaussianDistributionCfg(init_std= 1.0, std_type= "log"),
        )

    critic: MLPModelCfg = MLPModelCfg(
        class_name= "MLPModel",
        hidden_dims= [512, 256, 128],
        activation= "elu",
        obs_normalization= True,
        distribution_cfg= RslRlMLPModelCfg.GaussianDistributionCfg(init_std= 1.0, std_type= "log"),
        )

    encoder: EncoderMLPModelCfg = EncoderMLPModelCfg(
        class_name= "EncoderMLPModel",
        is_mlp_encoder= False, # Whether to use a MLP encoder. Defaults to True.
        hidden_dims= [256, 128],
        activation= "elu",
        output_dim= 3,
        )
    symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=True,
        mirror_loss_coeff=0.24,
        data_augmentation_func=wl_rough.compute_symmetric_states
    ),

    algorithm = EncoderRslRlPpoAlgorithmCfg(
        class_name= "PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        est_learning_rate=1.e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # optimizer= "adam", # ["adam", "adamw", "sgd", "rmsprop"]
        normalize_advantage_per_mini_batch= False,    # Whether to normalize the advantage per mini-batch. Defaults to False.
        # share_cnn_encoders= False,                    # Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False.
        rnd_cfg= None,                                # The RND configuration. Defaults to None, in which case RND is not used.
        symmetry_cfg= None,                           # The symmetry configuration. Defaults to None, in which case symmetry is not used.
    )
