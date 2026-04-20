"""
Autoresearch pretraining script. Single-GPU, single-file.
A short training run on an environment we wish to solve. Note that the real training run, after hyperparameter optimization, will be much longer.
Usage: python train.py --env-runners 60 --batch-size 65536 --minibatch-size 8192
"""
import time
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig, ActorCriticEncoderConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.encoder import TorchActorCriticEncoder
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.tune.registry import register_env

from constraints import TIME_BUDGET, ENV_CONFIG
from environments.SW_1v1_env_singleplayer import SW_1v1_env_singleplayer # Our testing environment
from classes.repeated_space import RepeatedCustom
from classes.batched_critic_ppo import BatchedCriticPPOLearner

import argparse

parser  = argparse.ArgumentParser()
parser.add_argument('--env-runners', type=int, default=60)
parser.add_argument("--batch-size", type=int, default=65536)
parser.add_argument("--minibatch-size", type=int, default=8192)
parser.add_argument("--critic-batch-size", type=int, default=32768) # Just for avoiding OOM issues
args = parser.parse_args()

#####
##### Model architecture; configured for use with RLlib
#####

CRITIC_ONLY = "CRITIC_ONLY" # Environment dynamics that the actor encoder should discard. Not used in our experiment.

class SimpleTransformerLayer(nn.Module): # A simple implementation of a transformer layer
    def __init__(self, emb_dim, heads, h_dim=2048, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.mha = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm_attn = torch.nn.LayerNorm(emb_dim)
        self.norm_ff = torch.nn.LayerNorm(emb_dim)
        self.residual = nn.Sequential(
            nn.Linear(emb_dim, h_dim),
            nn.GELU(), # Apparently just plain better than ReLU here.
            nn.Dropout(dropout),
            nn.Linear(h_dim, emb_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x, src_key_padding_mask):
        x_attn, _ = self.mha(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm_attn(x_attn + x)
        x_ff = self.residual(x)
        x = self.norm_ff(x_ff + x)
        return x

class AttentionEncoder(TorchModel, Encoder):
    """
    An Encoder that takes a Dict of multiple spaces, including Discrete, Box, and Repeated, and uses an attention layer to convert this variable-length input into a fixed-length featurized learned representation.
    """

    def __init__(self, config, is_critic):
        try:
            super().__init__(config)
            if (not is_critic): # Is this an actor/shared encoder or a critic encoder?
                self.is_critic_encoder = False;
            else:
                self.is_critic_encoder = True
            self.observation_space = config.observation_space
            self.emb_dim = config.emb_dim
            self.attn_layers = config.attn_layers
            self.use_deepset = getattr(config, 'use_deepset', False)
            if self.use_deepset:
                self.entity_mlp = nn.Sequential(
                    nn.Linear(self.emb_dim, config.attn_ff_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.attn_ff_dim, self.emb_dim),
                    nn.Dropout(config.dropout),
                )
                self.entity_norm = nn.LayerNorm(self.emb_dim)
            else:
                mhas = []
                for _ in range(self.attn_layers):
                    mhas.append(SimpleTransformerLayer(self.emb_dim, 4, h_dim=config.attn_ff_dim, dropout=config.dropout))
                self.mha = nn.ModuleList(mhas)
            # Can just run a bunch of these in sequence, they are self-contained.
            # Set up embedding layers for each element in our observation
            embs = {}
            for n, s in self.observation_space.spaces.items():
                if (CRITIC_ONLY in n and (not self.is_critic_encoder)):
                    continue # Ignore critic only information
                if type(s) is RepeatedCustom:
                    s = s.child_space  # embed layer applies to child space
                if type(s) is Box:
                    embs[n] = nn.Linear(s.shape[0], self.emb_dim)
                elif type(s) is Discrete:
                    embs[n] = nn.Embedding(s.n, self.emb_dim)
                    # By default, nn.Embedding has a much wider weight distribution than Linear.
                    # Hopefully this makes it easier to adapt to. Failing this, we freeze the other weights until a stable embedding is learned.
                    nn.init.uniform_(embs[n].weight, a=-.01, b=.01)
                else:
                    raise Exception("Unsupported observation subspace")
            self.embs = nn.ModuleDict(embs)
        except Exception as e:
            print("Exception when building AttentionEncoder:")
            print(e)
            raise e

    def _forward(self, input_dict, **kwargs):
        obs = input_dict[Columns.OBS]
        # The original space we mapped from.
        obs_s = self.observation_space
        embeddings = []
        masks = []
        for s in obs.keys():
            if (CRITIC_ONLY in s and (not self.is_critic_encoder)):
                continue # Ignore critic only information
            v = obs[s]
            v_s = obs_s[s]
            if (type(v_s) is Discrete):
                v = v.unsqueeze(-1) # Reshape for processing: [batch] -> [batch, 1]
            elif (len(v.shape)==1):
                v = v.unsqueeze(0)  # Add batch dimension: [feature] -> [1, feature]
            if type(v_s) is RepeatedCustom:
                v, mask = v_s.decode_obs(v)
                max_len = int(mask.sum(dim=1).max().item())
                if (max_len==0):    # Skip repeated spaces with no items
                    continue
                v = v[:,:max_len,:] # Improve efficiency for 'sparse' repeated spaces.
                mask = mask[:,:max_len]
            elif type(v_s) is Box:
                mask = torch.ones((v.shape[0], 1)).to(
                    v.device
                ) # Fixed elements are always there
                v = v.unsqueeze(1) # Add sequence length dimension after batch
            elif type(v_s) is Discrete: # nn.Embedding adds the sequence length dimension automatically
                mask = torch.ones((v.shape[0], 1)).to(
                    v.device
                ) # Fixed elements are always there
            embedded = self.embs[s](v)
            embeddings.append(embedded)
            masks.append(mask)
        x = torch.concatenate(embeddings, dim=1)  # batch_size, seq_len, unit_size
        mask = torch.concatenate(masks, dim=1)  # batch_size, seq_len
        if self.use_deepset:
            x = self.entity_norm(x + self.entity_mlp(x))
        else:
            for i in range(self.attn_layers):
                layer = self.mha[i]
                x = layer(x, src_key_padding_mask=(1-mask))
        # Masked mean-pooling.
        mask = mask.unsqueeze(dim=2)
        x = x * mask
        x = x.mean(dim=1) * mask.shape[1] / mask.sum(dim=1)
        return {ENCODER_OUT: x}


class AttentionEncoderConfig(ModelConfig):
    def __init__(self, observation_space, **kwargs):
        self.observation_space = observation_space
        self.emb_dim = kwargs["model_config_dict"]["attention_emb_dim"]
        self.attn_ff_dim = kwargs["model_config_dict"]["attn_ff_dim"]
        self.attn_layers = kwargs["model_config_dict"]["attn_layers"]
        self.dropout = kwargs["model_config_dict"].get("dropout", 0.1)
        self.use_deepset = kwargs["model_config_dict"].get("use_deepset", False)
        self.output_dims = (self.emb_dim,)

    def build(self, framework, is_critic=False):
        return AttentionEncoder(self, is_critic)

    def output_dims(self):
        return self.output_dims
        
class PartialObservabilityEncoder(TorchActorCriticEncoder):
    def __init__(self, config: ModelConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        if config.shared:
            self.encoder = config.base_encoder_config.build(framework=self.framework, is_critic=False)
        else:
            self.actor_encoder = config.base_encoder_config.build(
                framework=self.framework,
                is_critic=False
            )
            self.critic_encoder = None
            if not config.inference_only:
                self.critic_encoder = config.base_encoder_config.build(
                    framework=self.framework,
                    is_critic=True
                )
        
class PartialObservabilityEncoderConfig(ActorCriticEncoderConfig):
    """
    A modification of ActorCriticEncoderConfig that supports partially-observable environments with an omniscient critic.
    """
    def build(self, framework: str = "torch") -> "Encoder":
        return PartialObservabilityEncoder(self)

class AttentionPPOCatalog(PPOCatalog):
    """
    A special PPO catalog producing an encoder that handles dictionaries of (potentially Repeated) action spaces in the same manner as https://arxiv.org/abs/1909.07528.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
        self.actor_critic_encoder_config = PartialObservabilityEncoderConfig(
            base_encoder_config=self._encoder_config,
            shared=self._model_config_dict["vf_share_layers"],
        ) # Informs the critic encoder that it's the critic encoder.
        # Temporary code to enable use of Leaky ReLU activation functional
        self.override_activation_fn = self._model_config_dict.get('override_activation_fn', False)
        # Temporary code for adding layer normalization to the head. Will remove when fix is added to master branch.
        # https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo_catalog.py
        from ray.rllib.core.models.configs import MLPHeadConfig
        self.vf_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.pi_and_vf_head_hiddens,
            hidden_layer_activation=self.pi_and_vf_head_activation,
            hidden_layer_use_layernorm=self._model_config_dict.get(
                "head_fcnet_use_layernorm", False
            ),
            output_layer_activation="linear",
            output_layer_dim=1,
        )
        
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        **kwargs,
    ):
        return AttentionEncoderConfig(observation_space, **kwargs)
        
    # Temporary code for adding layer normalization to the head.
    # https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo_catalog.py
    def build_pi_head(self, framework: str):
        from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian
        from ray.rllib.core.models.configs import (
            FreeLogStdMLPHeadConfig,
            MLPHeadConfig,
        )
        # Get action_distribution_cls to find out about the output dimension for pi_head
        action_distribution_cls = self.get_action_dist_cls(framework=framework)
        if self._model_config_dict["free_log_std"]:
            _check_if_diag_gaussian(
                action_distribution_cls=action_distribution_cls, framework=framework
            )
            is_diag_gaussian = True
        else:
            is_diag_gaussian = _check_if_diag_gaussian(
                action_distribution_cls=action_distribution_cls,
                framework=framework,
                no_error=True,
            )
        required_output_dim = action_distribution_cls.required_input_dim(
            space=self.action_space, model_config=self._model_config_dict
        )
        # Now that we have the action dist class and number of outputs, we can define
        # our pi-config and build the pi head.
        pi_head_config_class = (
            FreeLogStdMLPHeadConfig
            if self._model_config_dict["free_log_std"]
            else MLPHeadConfig
        )
        self.pi_head_config = pi_head_config_class(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.pi_and_vf_head_hiddens,
            hidden_layer_activation=self.pi_and_vf_head_activation,
            hidden_layer_use_layernorm=self._model_config_dict.get(
                "head_fcnet_use_layernorm", False
            ),
            output_layer_dim=required_output_dim,
            output_layer_activation="linear",
            clip_log_std=is_diag_gaussian,
            log_std_clip_param=self._model_config_dict.get("log_std_clip_param", 20),
        )
        
        pi_head = self.pi_head_config.build(framework=framework)
        # Temporary code until RLlib adds LeakyReLU to its supported activation functions
        if (self.override_activation_fn):
            from ray.rllib.models.utils import get_activation_fn
            old_act_class = get_activation_fn(self.pi_and_vf_head_activation, framework='torch')
            new_layers = []
            for layer in pi_head.net.mlp:
                if isinstance(layer, old_act_class):
                    # Replace with LeakyReLU (adjust negative_slope if desired)
                    new_layers.append(nn.LeakyReLU())
                else:
                    new_layers.append(layer)
            pi_head.net.mlp = nn.Sequential(*new_layers)
        return pi_head
        
    def build_vf_head(self, framework: str):
        # Temporary code until RLlib adds LeakyReLU to its supported activation functions
        from ray.rllib.models.utils import get_activation_fn
        vf_head = self.vf_head_config.build(framework=framework)
        if (self.override_activation_fn):
            old_act_class = get_activation_fn(self.pi_and_vf_head_activation, framework='torch')
            new_layers = []
            for layer in vf_head.net.mlp:
                if isinstance(layer, old_act_class):
                    # Replace with LeakyReLU (adjust negative_slope if desired)
                    new_layers.append(nn.LeakyReLU())
                else:
                    new_layers.append(layer)
            vf_head.net.mlp = nn.Sequential(*new_layers)
        return vf_head 

#####
##### Run Training
#####
t_start = time.time()

target_env = SW_1v1_env_singleplayer
register_env("env", lambda cfg: target_env(cfg))

# Configure PPO
config = (
    PPOConfig()
    .env_runners(
        num_env_runners=args.env_runners,
        num_envs_per_env_runner=8,
    )
    .learners(
        num_gpus_per_learner=1,
    )
    .environment(
        env="env",
        env_config=ENV_CONFIG,
    )
    .training(
        lr=1e-4,
        gamma=0.999,
        lambda_=0.8,
        vf_clip_param=40,
        entropy_coeff=0.0,
        use_kl_loss=False,
        train_batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        learner_class=BatchedCriticPPOLearner,
        learner_config_dict={
            'critic_batch_size': args.critic_batch_size, # Just to avoid OOM; not a hyperparameter
        },
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            catalog_class=AttentionPPOCatalog,
            model_config={
                "attention_emb_dim": 128,
                "attn_ff_dim": 1024,
                "head_fcnet_hiddens": tuple([256,256]),
                "head_fcnet_activation": "relu",
                "vf_share_layers": True,
                "head_fcnet_use_layernorm": True,
                "attn_layers": 1,
                "use_deepset": True,
                "dropout": 0.1,
                
                "head_fcnet_activation": "relu",
                "override_activation_fn": True,
            },
        )
    )
    .debugging(seed=42)
)

algo = config.build_algo()

t_start_training = time.time()

step = 0
scores = []
while (True):
    step += 1
    results = algo.train()
    train_score = results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
    print(f"TRAIN: {train_score:.2f} time={time.time()-t_start_training:.1f}")
    scores.append(train_score)
    if time.time() - t_start_training >= TIME_BUDGET:
        print(f"Time budget reached at {step} iters.")
        break
        
t_end = time.time()
total_training_time = t_end - t_start_training

print("---")
print(f"AULC_score:       {np.mean(scores):.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {sum(p.numel() for p in algo.get_module().parameters()) / 1e6:.1f}")