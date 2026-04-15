import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw
import pygame

from classes.repeated_space import RepeatedCustom

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship, ego_pt, wrap
      
class Dummy_Ship(Ship):
    def update(self, missiles, speed, grav_multiplier, target_loc):
        if (self.stored_missiles > 0):
            # Rotate towards player
            vec_diff = target_loc-self.pos
            # wrap vec_diff
            wrap(vec_diff)
            ang_diff = (np.arctan2(-vec_diff[1],vec_diff[0]) * 180/np.pi - self.ang)%360
            if (ang_diff > 180): 
                action = 2 # turn left; positive angle greater than pi
            elif (ang_diff < -180):
                action = 1 # turn right; negative angle less than -pi
            elif (ang_diff > 0):
                action = 1 # turn right; positive angle less than pi
            else:
                action = 2
            if (action==1):
              self.ang += SHIP_TURN_RATE * speed
            elif (action==2):
              self.ang -= SHIP_TURN_RATE * speed
            self.updateAngUV()
            # Shoot
            if (self.stored_missiles > 0 and self.reloadTime <= 0 
                and (vec_diff**2).sum()**.5 < MISSILE_LIFE*MISSILE_VEL*1.5):
                m = Missile(self.pos + self.angUV * self.size, self.vel + self.angUV * MISSILE_VEL)
                missiles.append(m)
                self.stored_missiles -= 1
                self.reloadTime = MISSILE_RELOAD_TIME
            else:
                self.reloadTime =  max(self.reloadTime-speed,0)
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

class SW_1v1_env_singleplayer(gym.Env):
    def __init__(self, env_config={}):
        super().__init__()
        self.agents = self.possible_agents = [0]
        # nop/thrust, nop/left/right, npo/shoot
        self.egocentric = env_config.get('egocentric', True) # Egocentric coordinates are just better.
        self.render_egocentric = env_config.get('render_egocentric', False)
        self.maxTime = env_config.get('ep_length', DEFAULT_MAX_TIME)
        # Later on, could replace speed's multiplier with a for loop for better fidelity/robustness.
        # (Sample actions every K steps, shoot once, every other command gets repeated)
        self.speed = env_config.get('speed', 1.0)
        self.size = env_config.get('render_size', DEFAULT_RENDER_SIZE)
        # Gravity multiplier for curriculum learning
        self.grav_multiplier = self.true_grav_multiplier = env_config.get('grav_multiplier', 1.0)
        # Target size multiplier for curriculum learning (multiply radius to scale area linearly)
        self.size_multiplier = self.true_size_multiplier = env_config.get('size_multiplier', 1.0)
        # Target speed multiplier. A proportion of the stable orbital velocity
        self.target_speed = self.true_target_speed = env_config.get('target_speed', 0.0)
        self.elliptical = env_config.get('elliptical', True)
        self.target_ammo = self.true_target_ammo = env_config.get('target_ammo', 0.0)
        # Randomize the curriculum?
        self.probabilistic = env_config.get('probabilistic_difficulty', False)
        # Rendering
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'
        # Observation spaces; fixed and variable
        self.augment_obs = env_config.get('aug_obs', self.egocentric)
        assert (self.egocentric==True or self.augment_obs==False), "Augmentated observations supported for ego. obs. only"
        missile_space = Box(-1,1,shape=(Missile.REPR_SIZE+Missile.AUG_DIM*self.augment_obs,))
        self.missile_space = RepeatedCustom(missile_space, NUM_MISSILES)
        obs_space = {
            "self": Box(-1,1,shape=(Ship.REPR_SIZE+Ship.ALL_AUG_DIM*self.augment_obs,)), 
            "opponent": Box(-1,1,shape=(Ship.REPR_SIZE+Ship.OTHER_AUG_DIM*self.augment_obs,)),
            "missiles_friendly": self.missile_space, # Friendly missiles
            "missiles_hostile": self.missile_space # Hostile missiles
        }
        self.observation_space = Dict(obs_space)
        self.action_space = MultiDiscrete([2,3,2])
        
    def get_obs(self):
        ego = self.playerShips[0] if self.egocentric else None
        obs = {
            "self": self.playerShips[0].get_obs(ego, self.augment_obs),
            "opponent": self.playerShips[1].get_obs(ego, self.augment_obs),
            "missiles_friendly": self.missile_space.encode_obs([m.get_obs(ego, self.augment_obs) for m in self.missiles]),
            "missiles_hostile":  self.missile_space.encode_obs([m.get_obs(ego, self.augment_obs) for m in self.opponent_missiles]),
        }
        return obs
        
    def get_keymap(self): # Set multidiscrete 
        return {0: {pygame.K_UP: (0,1,False), # Action, Value, hold_disallowed (qol)
                pygame.K_LEFT: (1,1,False), pygame.K_RIGHT: (1,2,False),
                pygame.K_DOWN: (2,1,False),
                "default": 0},
                }
                
    def new_target_position(self):
        target = self.playerShips[1]
        if (self.target_speed == 0):
            position = self.rng.uniform(-WRAP_BOUND,WRAP_BOUND, (2,))
        else:
            position = self.rng.uniform(-WRAP_BOUND,WRAP_BOUND, (2,))
            r_min = PLAYER_SIZE * 2 + STAR_SIZE
            r = max(r_min, self.rng.uniform(0, 1)**.5 * WRAP_BOUND)
            p_ang = self.rng.uniform(0, 2*np.pi)
            position = np.array([np.cos(p_ang), np.sin(p_ang)]) * r
            dist_zero = (position**2).sum()**.5
            # Spawn it somewhere the player isn't
            player_pos = self.playerShips[0].pos
            disp = (position-player_pos)
            bounds_dist = MISSILE_LIFE*MISSILE_VEL*1.2
            dist_from_valid = bounds_dist - (disp**2).sum()**.5
            if (dist_from_valid > 0):
                # Shunt it the extra distance away
                position = player_pos + disp / (1 - dist_from_valid / bounds_dist)
                # wrap it (around a circle) if it exceeds the bounds.
                pos_r = (position**2).sum()**.5
                if (pos_r > WRAP_BOUND):
                    position = position * -(2*WRAP_BOUND-pos_r) / pos_r
            # Set velocity (perpendicular to angle to star)
            g = GRAV_CONST
            if (not self.elliptical):
                v_magnitude = (g/r)**.5 # For a circular orbit (v^2 = GM/r; g=GM)
            else: #   eccentricity = (r_a - r_p) / (r_a + r_p); r_a = r
                r_p = self.rng.uniform(low=max(r_min, r / 2), high=r)
                ecc = (r - r_p) / (r + r_p)
                # velocity at apocentre is sqrt((1-e) * GM / ((1+e) * r_a))
                v_magnitude = ((1-ecc) * g / ((1+ecc)*r))**.5
            v_magnitude *= self.target_speed
            v_angle = np.arctan2(position[1],position[0]) + np.pi/2 * np.sign(self.rng.random() - 0.5)
            target.vel = np.array([np.cos(v_angle), np.sin(v_angle)]) * v_magnitude
        target.pos = position
        if (self.target_ammo != 0):
            target.stored_missiles = int(NUM_MISSILES * self.target_ammo)
            target.ang =  self.rng.uniform(0,360) # Random initial angle
            target.updateAngUV()
        else:
            target.stored_missiles = 0
    
    def reset(self, seed=None, options={}):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif not hasattr(self, 'rng'):
            self.rng = np.random.default_rng()
        if (self.probabilistic == True):
            self.grav_multiplier = np.clip(self.true_grav_multiplier + self.rng.normal(0,0.1), 0, 1) # [0,1], sd of 0.1
            self.size_multiplier = np.clip(self.true_size_multiplier + self.rng.normal(0,1), 1, 10)  # [1,10], sd of 1
            self.target_speed = np.clip(self.true_target_speed + self.rng.normal(0,0.1), 0, 1)       # [0,1], sd of 0.1
            self.target_ammo = np.clip(self.true_target_ammo + self.rng.normal(0,0.1), 0, 1)       # [0,1], sd of 0.1
        self.playerShips = [
            Ship(np.array([-.5, .5]), 90.),
            Dummy_Ship(np.array([0.,0.]),0.,PLAYER_SIZE*self.size_multiplier)
        ]
        self.new_target_position()
        self.missiles = [] # x, y, vx, vy
        self.opponent_missiles = []
        self.time = 0
        self.terminated = False # for rendering purposes
        return self.get_obs(), {}
        
    def step(self, action):
        self.rewards = 0
        self.time += 1 * self.speed
        # Thrust is acc times anguv
        ship = self.playerShips[0]
        ship.update(action, self.missiles, self.speed, grav_multiplier=self.grav_multiplier)
        if (np.linalg.norm(ship.pos, 2) < PLAYER_SIZE + STAR_SIZE):
            self.terminated = True;
            self.rewards = -1
        # Update the dummy ship
        target = self.playerShips[1]
        if ((self.target_speed != 0) or (target.stored_missiles != 0)):
            target.update(self.opponent_missiles, self.speed, 
                grav_multiplier=self.grav_multiplier*self.target_speed, target_loc=ship.pos)
            if (np.linalg.norm(target.pos, 2) < PLAYER_SIZE + STAR_SIZE):
                self.new_target_position() # If it crashes, respawn it
        # Update missiles
        ms = [self.missiles, self.opponent_missiles]
        for m in ms:
            for i in reversed(range(len(m))):
                si,d = m[i].update(self.playerShips, self.speed) # Return hit_obj
                if (d):
                    del m[i]
                if (si != -1):
                    if (si == 0):
                        self.terminated = True
                        self.rewards = -1
                    else:
                        self.rewards = 1
                        self.new_target_position()
        truncated = (self.time >= self.maxTime)
        return self.get_obs(), self.rewards, self.terminated, truncated, {}
        
    def render(self): # Display the environment state
        ego = self.playerShips[0] if self.render_egocentric else None
        dim = self.size
        hdim=dim/2
        ssz = STAR_SIZE * dim
        msz = 3
        img = Image.new('RGB', (dim, dim), color='black')
        draw = ImageDraw.Draw(img)
        # Draw the star (drawn by player ship if egocentric rendering is active)
        if (ego is None):
            for i in range(2):
                ss = np.random.uniform(-1.0, 1.0, 2) * ssz/2
                draw.line((hdim+ss[0], hdim+ss[1], hdim-ss[0], hdim-ss[1]), fill='white', width=1)
            # Draw the wrapping radius 
            #draw.ellipse((0, 0, dim, dim), outline='white')
            rs = (1-WRAP_BOUND) * hdim
            draw.rectangle((rs,rs,dim-rs,dim-rs), outline='white')
        # Draw the player
        ship = self.playerShips[0]
        ship.render(draw, dim, hdim, ssz, self.terminated, reward=self.rewards[0], ego=ego)
        # Draw the target
        target = self.playerShips[1]
        target.render(draw, hdim, ego)
        # Draw the missiles
        for m in self.missiles:
            m.render(draw, hdim, msz, self.speed, ego=ego)
        for m in self.opponent_missiles:
            m.render(draw, hdim, msz, self.speed, ego=ego, c="orange")
        # Rotate 90 degrees for easier viewing when using egocentric rendering
        if (self.egocentric):
            img = img.rotate(90)
            draw = ImageDraw.Draw(img)
        # Draw the time
        draw.text((10, img.height-60), f"Time: {self.time}/{self.maxTime}", (255, 255, 255))
        return img
