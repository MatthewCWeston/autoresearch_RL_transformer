# This class is not used at all in our testing configuration. There is no need to view or edit it.
import gymnasium as gym
from ray.rllib.utils.annotations import PublicAPI
import numpy as np


@PublicAPI
class RepeatedCustom(gym.spaces.Box):
    """
      An ad-hoc fix for RLlib's Repeated gym space. Encodes and decodes lists
      of its target space in the format:
      [one-hot mask + concatenated observations]
    """

    def __init__(self, child_space: gym.spaces.Box, max_len: int):
        assert len(child_space.shape) == 1, "Child space must be a flat Box."
        self.child_space = child_space
        self.max_len = max_len
        self.dtype = child_space.dtype
        low = np.concatenate([np.zeros(max_len,dtype=self.dtype),np.repeat(child_space.low, max_len)])
        high = np.concatenate([np.ones(max_len,dtype=self.dtype),np.repeat(child_space.high, max_len)])
        shape = (child_space.shape[0]*max_len+max_len,)
        self.practical_space = gym.spaces.Box(
            low=low, high=high,shape=shape,dtype=self.dtype
        )
        super().__init__(low=low,high=high,shape=shape,dtype=self.dtype)

    def encode_obs(self, obs):
      mask = np.zeros(self.max_len, dtype=self.dtype)
      mask[:len(obs)] = 1
      obs = np.concatenate([mask]+obs)
      encoded = np.zeros(self.max_len + self.max_len * self.child_space.shape[0],dtype=self.dtype)
      encoded[:len(obs)] = obs
      return encoded

    def decode_obs(self, obs):
      mask = obs[:,:self.max_len]
      N = 1 if len(obs.shape)==1 else obs.shape[0] # batch size
      obs = obs[:, self.max_len:].reshape(
          (N, self.max_len, self.child_space.shape[0])
      ) # batch, seq, subspace
      return obs, mask

    def sample(self):
        sampled_input = [
            self.child_space.sample()
            for _ in range(self.np_random.integers(1, self.max_len + 1))
        ]
        return self.practical_space.sample()#self.encode_obs(sampled_input)

    def contains(self, x):
        return self.practical_space.contains(x)

    def __repr__(self):
        return "RepeatedCustom({}, {})".format(self.child_space, self.max_len)