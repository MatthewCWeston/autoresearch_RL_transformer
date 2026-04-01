"""
Autoresearch pretraining script. Single-GPU, single-file.
Simplified from our target environment.
Usage: uv run train.py
"""
import time
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)

import torch
from torch import nn
import torch.nn.functional as F

from env import SW_lead_target
from misc import RepeatedCustom, TIME_BUDGET

#####
##### Custom architecture
#####

class SimpleTransformerLayer(nn.Module):
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
            # Use an attention layer to reduce observations to a fixed length
            mhas = []
            for _ in range(self.attn_layers):
                mhas.append(SimpleTransformerLayer(self.emb_dim, 4, h_dim=config.attn_ff_dim, dropout=config.dropout))
            self.mha = nn.ModuleList(mhas)
            # Can just run a bunch of these in sequence, they are self-contained.
            # Set up embedding layers for each element in our observation
            embs = {}
            for n, s in self.observation_space.spaces.items():
                if type(s) is RepeatedCustom:
                    s = s.child_space  # embed layer applies to child space
                if type(s) is Box:
                    embs[n] = nn.Linear(s.shape[0], self.emb_dim)
                elif type(s) is Discrete:
                    embs[n] = nn.Embedding(s.n, self.emb_dim)
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
        # All entities have embeddings. Apply masked residual self-attention and then mean-pool.
        x = torch.concatenate(embeddings, dim=1)  # batch_size, seq_len, unit_size
        mask = torch.concatenate(masks, dim=1)  # batch_size, seq_len
        for i in range(self.attn_layers):
            layer = self.mha[i]
            x = layer(x, src_key_padding_mask=(1-mask))
        # Masked mean-pooling.
        mask = mask.unsqueeze(dim=2)
        x = x * mask  # Mask x to exclude nonexistent entries from mean pool op
        x = x.mean(dim=1) * mask.shape[1] / mask.sum(dim=1)  # Adjust mean
        return {ENCODER_OUT: x}


class AttentionEncoderConfig(ModelConfig):
    def __init__(self, observation_space, **kwargs):
        self.observation_space = observation_space
        self.emb_dim = kwargs["model_config_dict"]["attention_emb_dim"]
        self.attn_ff_dim = kwargs["model_config_dict"]["attn_ff_dim"]
        self.attn_layers = kwargs["model_config_dict"]["attn_layers"]
        self.dropout = kwargs["model_config_dict"].get("dropout", 0.1)
        self.output_dims = (self.emb_dim,)

    def build(self, framework, is_critic=False):
        return AttentionEncoder(self, is_critic)

    def output_dims(self):
        return self.output_dims

class AttentionPPOCatalog(PPOCatalog):
    """
    A special PPO catalog producing an encoder that handles dictionaries of (potentially Repeated) action spaces in the same manner as https://arxiv.org/abs/1909.07528.
    """
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        **kwargs,
    ):
        return AttentionEncoderConfig(observation_space, **kwargs)

#####
##### Run Training
#####
t_start = time.time()

# Configure PPO
env_runners = 8
config = (
    PPOConfig()
    .env_runners(
        num_env_runners=env_runners,
        num_envs_per_env_runner=8,
    )
    .environment(
        env=SW_lead_target
    )
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.01,
        train_batch_size=8192,
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            catalog_class=AttentionPPOCatalog,
            model_config={
                "attention_emb_dim": 32,
                "attn_ff_dim": 128,
                "head_fcnet_hiddens": tuple([64,64]),
                "head_fcnet_activation": "relu",
                "vf_share_layers": False,
                "head_fcnet_use_layernorm": True,
                "attn_layers": 1,
                "dropout": 0.1,
            },
        )
    )
    .evaluation(
		evaluation_num_env_runners=1,
		evaluation_interval=0, # No evaluations while training
		evaluation_duration=100,
        evaluation_duration_unit="episodes",
	)
)

algo = config.build_algo()

num_iters = 200
t_start_training = time.time()

for step in tqdm(range(num_iters)):
  results = algo.train()
  print(f"TRAIN: {results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}:.2f")
  if time.time() - t_start_training >= TIME_BUDGET:
      print(f"Time budget reached at {step+1} iters.")
      break
total_training_time = time.time() - t_start_training
results = algo.evaluate()
eval_score = results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
print(f"EVALUATION: {eval_score}")

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start

print("---")
print(f"eval_score:          {eval_score:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
print(f"num_steps:        {step+1}")
print(f"num_params_M:     {sum(p.numel() for p in algo.get_module().parameters()) / 1e6:.1f}")