'''
Note that the environment you are operating on is a simplified test-case meant to be challenging in a similar way to a separate environment on which your final architecture will be deployed. The larger environment will, as an example, involve reasoning about variable-length lists of moving targets, and maneuvering to avoid a variable-length list of opposing projectiles. The base architecture I have provided can handle these things, and it should still be able to do so when the automatic research process is complete. Moreover, you should make improvements to the architecture and training process that you reasonably expect would generalize to the larger environment.
'''
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw
import pygame

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship, ego_pt, wrap
      
class Dummy_Ship(Ship):
    def update(self, missiles, speed, grav_multiplier, target_loc):
        self.updateAngUV()
        # Update position
        self.pos += self.vel * speed
        self.vel = np.clip(self.vel, -1.0, 1.0)
        wrap(self.pos)
        # Apply force of gravity. GMm can be treated as a single constant.
        self.vel -= (self.pos * GRAV_CONST / (self.pos[0]**2 + self.pos[1]**2)** 1.5) * speed * grav_multiplier
        
    def render(self, draw, hdim, ego=None):
        super().render(draw, hdim*2, hdim, self.size, False, ego=ego)
        psz = self.size * hdim*2
        if (ego is None):
            pos = self.pos
        else:
            pos = self.get_obs(ego=ego)[:2]
        p = (pos+1) * hdim
        draw.ellipse((p[0]-psz/2, p[1]-psz/2, p[0]+psz/2, p[1]+psz/2), outline='yellow')
        if ((self.vel!=0).any()):
            vel = self.vel * psz*hdim*5
            draw.line([p[0],p[1], p[0]+vel[0],p[1]+vel[1]], width=1, fill='cyan')
            draw.line([p[0],p[1], hdim,hdim], width=1, fill='green')
   

class SW_lead_target(MultiAgentEnv):
    def __init__(self, env_config={}):
        super().__init__()
        self.agents = self.possible_agents = [0]
        # nop/thrust, nop/left/right, npo/shoot
        self.egocentric = env_config.get('egocentric', True) # Egocentric coordinates are just better.
        self.render_egocentric = env_config.get('render_egocentric', False)
        self.maxTime = env_config.get('ep_length', DEFAULT_MAX_TIME)
        self.speed = env_config.get('speed', 1.0)
        self.size = env_config.get('render_size', DEFAULT_RENDER_SIZE)
        # Target size multiplier
        self.size_multiplier = self.true_size_multiplier = env_config.get('size_multiplier', 1.0)
        self.elliptical = env_config.get('elliptical', True)
        # Skip steps after missile launches (makes value learning easier, since nothing to do after launch)
        self.skip_after_launch = env_config.get('skip_after_launch', True)
        # Rendering
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'
        # Observation spaces; fixed and variable
        ship_space = Box(-1,1,shape=(Ship.REPR_SIZE,))
        self.missile_space = Box(-1,1,shape=(Missile.REPR_SIZE,))
        obs_space = {
            "self": ship_space, # my ship, enemy ship
            "opponent": ship_space,
            "missiles_friendly": self.missile_space, # Friendly missiles
        }
        self.observation_space = Dict(obs_space)
        self.action_space = MultiDiscrete([3,2])
        
    def get_keymap(self): # Set multidiscrete 
        return {0: {
                pygame.K_LEFT: (0,1,False),
                pygame.K_RIGHT: (0,2,False),
                pygame.K_DOWN: (1,1,False),
                "default": 0},
                }
        
    def get_obs(self):
        ego = self.playerShips[0] if self.egocentric else None
        m = self.missiles
        obs = {
            "self": self.playerShips[0].get_obs(ego),
            "opponent": self.playerShips[1].get_obs(ego),
            "missiles_friendly": m[0].get_obs(ego) if len(m) > 0 else np.zeros((Missile.REPR_SIZE,)),
        }
        return obs
                
    def new_target_position(self):
        target = self.playerShips[1]
        r = self.rng.uniform(0, 1) ** .5 * MISSILE_VEL * MISSILE_LIFE
        p_ang = self.rng.uniform(0, 2 * np.pi)
        position = np.array([np.cos(p_ang), np.sin(p_ang)]) * r
        g = GRAV_CONST
        if not self.elliptical:
            v_magnitude = (g / r) ** .5
        else:
            r_p = self.rng.uniform(low=max(PLAYER_SIZE * 2 + STAR_SIZE, r / 2), high=r)
            ecc = (r - r_p) / (r + r_p)
            v_magnitude = ((1 - ecc) * g / ((1 + ecc) * r)) ** .5
        v_angle = np.arctan2(position[1], position[0]) + np.pi / 2 * np.sign(self.rng.random() - 0.5)
        target.vel = np.array([np.cos(v_angle), np.sin(v_angle)]) * v_magnitude
        target.pos = position
    
    def reset(self, seed=None, options={}):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif not hasattr(self, 'rng'):
            self.rng = np.random.default_rng()
        self.playerShips = [
            Ship(np.array([1e-10, 0.]), 90.),
            Dummy_Ship(np.array([0., 0.]), 0., PLAYER_SIZE * self.size_multiplier)
        ]
        self.playerShips[0].stored_missiles = 1
        self.new_target_position()
        self.missiles = []
        self.time = 0
        self.terminated = False
        self.target_dots = []
        self.missile_dots = []
        return self.get_obs(), {}
     
    def step(self, actions):
        actions = [actions]
        self.rewards = {0:0}
        self.time += 1 * self.speed
        # Thrust is acc times anguv
        ship = self.playerShips[0]
        actions = [
            0, # Thrust (none)
            actions[0][0], # Turn (right or left only)
            actions[0][1]    # shoot (nop or fire)
        ]
        # If we have a missile already, loop until it resolves.
        target = self.playerShips[1]
        ms = self.missiles
        if (self.skip_after_launch):
            while(len(ms)>0):
                target.update([], self.speed, grav_multiplier=1, target_loc=ship.pos)
                si,d = ms[0].update(self.playerShips, self.speed)
                self.target_dots.append(target.pos.copy())
                self.missile_dots.append(ms[0].pos.copy())
                if (d):
                    del ms[0]
                if (si != -1):
                    self.terminated = True
                    self.rewards[0] = 1
        ship.update(actions, self.missiles, self.speed, grav_multiplier=0)
        # Update the dummy ship
        target.update([], self.speed, 
            grav_multiplier=1, target_loc=ship.pos)
        if (np.linalg.norm(target.pos, 2) < PLAYER_SIZE + STAR_SIZE):
            self.new_target_position() # If it crashes, respawn it
        # Update missiles
        for i in reversed(range(len(ms))):
            si,d = ms[i].update(self.playerShips, self.speed) # Return hit_obj
            if (d):
                del ms[i]
            if (si != -1):
                self.terminated = True
                self.rewards[0] = 1
        if (ship.stored_missiles == 0 and len(ms)==0):
            self.terminated = True # End environment if we missed.
        truncated = (self.time >= self.maxTime)
        return self.get_obs(), self.rewards[0], self.terminated, truncated, {}
        
    def render(self): # Display the environment state
        ego = self.playerShips[0] if self.render_egocentric else None
        dim = self.size
        hdim=dim/2
        msz = 3
        img = Image.new('RGB', (dim, dim), color='black')
        draw = ImageDraw.Draw(img)
        # Draw the star (drawn by player ship if egocentric rendering is active)
        if (ego is None):
            rs = (1-WRAP_BOUND) * hdim
            draw.rectangle((rs,rs,dim-rs,dim-rs), outline='white')
        # Draw the player
        ship = self.playerShips[0]
        ship.render(draw, dim, hdim, 1, self.terminated, ego=ego)
        # draw dots if env has ended
        for i, d in enumerate(self.target_dots):
            d = (d+1)*hdim
            draw.ellipse((d[0]-msz, d[1]-msz, d[0]+msz, d[1]+msz), fill=(255, 0, 0, int(255*i/len(self.target_dots))))
        for i, d in enumerate(self.missile_dots):
            d = (d+1)*hdim
            draw.ellipse((d[0]-msz, d[1]-msz, d[0]+msz, d[1]+msz), fill=(255, 255, 0, int(255*i/len(self.missile_dots))))
        # Draw the target
        target = self.playerShips[1]
        target.render(draw, hdim, ego)
        # Draw the missiles
        for m in self.missiles:
            m.render(draw, hdim, msz, self.speed, ego=ego)
        # Rotate 90 degrees for easier viewing when using egocentric rendering
        if (self.egocentric):
            img = img.rotate(90)
            draw = ImageDraw.Draw(img)
        # Draw the time
        draw.text((10, img.height-60), f"Time: {self.time}/{self.maxTime}", (255, 255, 255))
        return img