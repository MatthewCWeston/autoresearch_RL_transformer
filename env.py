# Contains the environment that we wish to solve. It involves some challenging orbital dynamics reasoning, and has a dictionary observation space that we wish to process using a transformer encoder. Note that this environment is a simplified reduction of a larger and more complex environment involving variable-length lists of moving ships and projectiles. The starting architecture I have provided can accommodate such an environment, and this compatibility should be preserved.
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw

from env_helpers import *

class SW_lead_target(gym.Env):
    def __init__(self, env_config={}):
        super().__init__()
        self.speed = env_config.get('speed', 5.0)
        ship_space = Box(-1,1,shape=(Ship.REPR_SIZE,))
        self.missile_space = Box(-1,1,shape=(Missile.REPR_SIZE,))
        obs_space = {
            "self": ship_space, # my ship, enemy ship
            "opponent": ship_space,
            "missiles_friendly": self.missile_space, # Friendly missiles
        }
        self.observation_space = Dict(obs_space)
        self.action_space = MultiDiscrete([3,2]) # nop/left/right, nop/shoot

    def get_obs(self):
        ego = self.playerShips[0]
        m = self.missiles
        obs = {
            "self": self.playerShips[0].get_obs(ego),
            "opponent": self.playerShips[1].get_obs(ego),
            "missiles_friendly": m[0].get_obs(ego) if len(m) > 0 else np.zeros((Missile.REPR_SIZE,), dtype=np.float32),
        }
        return obs

    def new_target_position(self):
        target = self.playerShips[1]
        position = np.random.uniform(-WRAP_BOUND,WRAP_BOUND, (2,))
        r = np.random.uniform(0, 1)**.5 * MISSILE_VEL * MISSILE_LIFE
        p_ang = np.random.uniform(0, 2*np.pi)
        position = np.array([np.cos(p_ang), np.sin(p_ang)]) * r
        # Set velocity (perpendicular to angle to star) so as to produce a usually-stable elliptical orbit
        g = GRAV_CONST
        r_p = np.random.uniform(low=max(PLAYER_SIZE*2+STAR_SIZE, r/2), high=r)
        ecc = (r - r_p) / (r + r_p)
        v_magnitude = ((1-ecc) * g / ((1+ecc)*r))**.5
        v_angle = np.arctan2(position[1],position[0]) + np.pi/2 * np.sign(np.random.rand()-0.5)
        target.vel = np.array([np.cos(v_angle), np.sin(v_angle)]) * v_magnitude
        target.pos = position

    def reset(self, seed=None, options={}):
        self.playerShips = [
            Ship(np.array([1e-10, 0.]), 90.),
            Dummy_Ship(np.array([0.,0.]),0.,PLAYER_SIZE)
        ]
        self.playerShips[0].stored_missiles = 1
        self.new_target_position()
        self.missiles = [] # x, y, vx, vy
        self.time = 0
        self.terminated = False # for rendering purposes
        return self.get_obs(), {}

    def step(self, actions):
        self.reward = 0
        self.time += 1 * self.speed
        # Thrust is acc times anguv
        ship = self.playerShips[0]
        actions = [
            actions[0], # Turn (right or left only)
            actions[1]    # shoot (nop or fire)
        ]
        # If we have a missile already, loop until it resolves.
        target = self.playerShips[1]
        ms = self.missiles
        while(len(ms)>0): # Skip to environment conclusion after the projectile is launched
            target.update(self.speed)
            si,d = ms[0].update(self.playerShips, self.speed)
            if (d):
                del ms[0]
            if (si != -1):
                self.terminated = True
                self.reward = 1
        ship.update(actions, self.missiles, self.speed)
        # Update the dummy ship
        target.update(self.speed)
        if (np.linalg.norm(target.pos, 2) < PLAYER_SIZE + STAR_SIZE):
            self.new_target_position() # If it crashes, respawn it
        if (ship.stored_missiles == 0 and len(ms)==0):
            self.terminated = True # End environment if we missed.
        truncated = (self.time >= 4096) # timeout
        return self.get_obs(), self.reward, self.terminated, truncated, {}