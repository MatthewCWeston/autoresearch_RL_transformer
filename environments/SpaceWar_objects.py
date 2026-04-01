import numpy as np
from environments.SpaceWar_constants import *
from environments.SpaceWar_constants import WRAP_BOUND

def wrap(p):
    ''' Wraps a point within a square centered on [0,0]'''
    for i in range(2):
        if (p[i]>WRAP_BOUND):
            p[i] -= 2*WRAP_BOUND
        elif (p[i]<-WRAP_BOUND):
            p[i] += 2*WRAP_BOUND
        
def rotate_pt(p, a):
    a = -a*np.pi/180
    x, y = p
    sa = np.sin(a)
    ca = np.cos(a)
    return np.array([x*ca-y*sa,y*ca+x*sa])
    
def ego_pt(p, ego): # egocentric coordinates, accounting for wrapping around a unit circle
    diff = p - ego.pos # relative position 
    # Wrap adjusted position by wrapping dist
    wrap(diff) # This line decides whether we wrap objects that are closer via wrapping
    # Adjust target position by angle.
    pos_adj = rotate_pt(diff, -ego.ang) # Rotate s.t. ego has angle zero
    return pos_adj
    
def gr_helper(p, a):
    a = a * np.pi/180
    m = np.tan(a)+1e-12 # slope from angle 
    i = p[1] - m * p[0] # y intercept from slope and point
    # for each edge, get the point of intersection. Horizontal edges are y=+-WB, vertical are x=+-WB
    pois = [
        ((WRAP_BOUND - i) / m, WRAP_BOUND), ((-WRAP_BOUND - i) / m, -WRAP_BOUND), # horizontal
        (WRAP_BOUND, m*WRAP_BOUND+i), (-WRAP_BOUND, m*-WRAP_BOUND+i), # vertical
        ]
    distances = [((x-p)**2).sum() for x in pois]
    signs = [np.sign(x[0]-p[0]) * np.sign(np.cos(a) * (-1 if ix < 2 else 1)) for ix, x in enumerate(pois)]
    ix = np.argmin(distances)
    return distances[ix]**.5 * signs[ix]
    
def get_raycasts(p, a):
    ''' Get the distances from a ship's front and side to the nearest edge''' 
    return [gr_helper(p, a), gr_helper(p, a-90)] # ignore second raycast for now

class Missile():
    REPR_SIZE = 5
    AUG_DIM = 4
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.life = self.maxLife = MISSILE_LIFE
    def update(self, ships, speed): # Returns hit_player, remove_self
        mp = self.pos
        mv = self.vel
        mp += mv * speed # Move missile
        for si in range(len(ships)): # Missile hits target
            if (np.linalg.norm(mp-ships[si].pos, 2) < ships[si].size):
                return si, True
        if (np.linalg.norm(mp, 2) < STAR_SIZE): # Missile hits star
            return -1, True
        wrap(mp)
        self.life -= 1 * speed
        return -1, (self.life<=0)
    def get_obs(self, ego, aug=False):
        if (ego is None):
            return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.life/self.maxLife])
        else:
            p = ego_pt(self.pos, ego)
            v = rotate_pt(self.vel, -ego.ang) # Rotate velocity w/r to removing ego's angle
            obs = np.concatenate([p, v, [self.life/self.maxLife]])
            if (aug): # Distance to observer, closing speed to observer
                dist = np.linalg.norm(p)
                closing_speed = np.dot(p / dist, (v - rotate_pt(ego.vel, -ego.ang))) / 2 # / 2 to normalize magnitude
                # Observer's bearing, from the perspective of the missile.
                ang_to_observer =  np.arctan2(p[0],p[1]) * 180/np.pi
                bearing_wrt_missile = rotate_pt(v / np.linalg.norm(v), -ang_to_observer) # angular UV of missile direction
                obs = np.concatenate([obs, bearing_wrt_missile, [dist, closing_speed]])
            return obs
    def render(self, draw, hdim, msz, envspeed, ego=None, c="yellow"):
        if (ego is None):
            pos, vel = self.pos, self.vel
        else:
            obs = self.get_obs(ego)
            pos, vel = obs[:2], obs[2:4]
        sz = msz if self.life > envspeed else msz*4
        m = (pos+1) * hdim
        draw.ellipse((m[0]-sz/2, m[1]-sz/2, m[0]+sz/2, m[1]+sz/2), outline=c, fill=c)
        # Draw a cyan line indicating velocity 
        if (ego is not None):
            v = vel*msz*hdim*3
            draw.line([m[0],m[1], m[0]+v[0],m[1]+v[1]], width=1, fill='cyan')

class Ship():
    REPR_SIZE = 8
    ALL_AUG_DIM = 4
    OTHER_AUG_DIM = ALL_AUG_DIM + 6
    def __init__(self, pos, ang, size=PLAYER_SIZE, vel=None):
        self.pos = pos
        self.ang = ang
        self.stored_missiles = NUM_MISSILES
        self.vel = vel if vel is not None else np.array([0.,0.])
        self.reloadTime = 0
        self.last_act = [0,0,0]
        self.size = size
        self.updateAngUV()
    def updateAngUV(self):
        a = self.ang*np.pi/180
        self.angUV = np.array([np.cos(a), -np.sin(a)])
    def update(self, action, missiles, speed, grav_multiplier=1.):
        self.last_act = action # for rendering
        # Take actions 
        if (action[0]==1):
          self.vel += SHIP_ACC * self.angUV * speed
        if (action[1]==1):
          self.ang += SHIP_TURN_RATE * speed
          self.updateAngUV()
        elif (action[1]==2):
          self.ang -= SHIP_TURN_RATE * speed
          self.updateAngUV()
        # Shoot 
        if (action[2]==1 and self.stored_missiles > 0 and self.reloadTime <= 0):
            m = Missile(self.pos + self.angUV * self.size, 
                self.vel + self.angUV * MISSILE_VEL)
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
    def get_obs(self, 
            ego=None, # The Ship associated with the observing agent
            aug=False,# Should we augment the observations with some engineered features?
            ):
        # pos, vel, angle unit vector, ammo remaining
        if (ego is None):
            return np.concatenate([self.pos, self.vel, self.angUV, 
            [self.stored_missiles / NUM_MISSILES, self.reloadTime / MISSILE_RELOAD_TIME]
            ])
        else:
            # Self: [star, vel, to_edges, missiles, reload time]
            # Other: [pos (relative to self), vel (rotated to self), auv (rotated to self), missiles, reload time]
            # We want to augment observations.
            # All: Radial and tangential velocity WRT star, distance to star, speed
            # Other: XY to star (rotated to self), bearing, distance to observer, closing speed to observer
            if (ego==self):
                p = rotate_pt(-self.pos, -ego.ang) # Location of star, adjusted for angle
                auv = get_raycasts(self.pos, self.ang) # Raycasts to arena boundary
            else:
                p = ego_pt(self.pos, ego)
                #auv = rotate_pt(self.angUV, -ego.ang) 
                # AngUV rotated by -angle of p, not by observer's facing
                ang_to_observer = np.arctan2(p[0],p[1]) * 180/np.pi
                auv = rotate_pt(self.angUV, -ang_to_observer) # Gets the 'bearing' of the observer in the eyes of this ship
            v = rotate_pt(self.vel, -ego.ang) # Rotate velocity w/r to removing ego's angle
            obs = np.concatenate([p, v, auv, 
                [self.stored_missiles / NUM_MISSILES, self.reloadTime / MISSILE_RELOAD_TIME]
            ])
            if (aug):
                star_dist = np.linalg.norm(self.pos) # Wrap bound permits a maximum of 1 distance from star
                star_anguv = -self.pos/star_dist # Angular unit vector of the direction to the star
                rad_vel = np.dot(star_anguv, self.vel) # Velocity component perpendicular to orbit
                tan_vel = np.dot(np.array([star_anguv[1],-star_anguv[0]]), self.vel) # Velocity component of orbit
                speed = np.linalg.norm(self.vel) # Speed
                aug_vec = [rad_vel, tan_vel, speed, star_dist]
                if (ego != self):
                    # Include X,Y distance to star, rotated WRT ego
                    star_coords = rotate_pt(-self.pos, -ego.ang)
                    # Bearing: Egocentric position of target, normalized by distance to target.
                    dist = np.linalg.norm(p)
                    opp_bearing = p / dist
                    # Speed at which distance is being closed to ego point
                    closing_speed = np.dot(opp_bearing, (v - rotate_pt(ego.vel, -ego.ang))) / 2
                    aug_vec.extend(np.concatenate([star_coords, opp_bearing, [dist, closing_speed]]))
                obs = np.concatenate([obs, aug_vec])
            return obs.clip(-1,1) # avoid rounding errors
    def render(self, draw,
                dim, hdim, ssz, terminated, ego=None, reward=0):
        ''' Render this object using draw, adjust for ego if needed.'''
        psz = self.size * dim
        if (ego is None):
            pos, vel, auv = self.pos, self.vel, self.angUV
            ang = self.ang
        else:
            adj = self.get_obs(ego)
            # Note AUV is instead the position of the nearest corner when ego==self
            pos, vel, auv = adj[:2],adj[2:4],adj[4:6]
            ang = self.ang-ego.ang
            if (ego==self):
                # Draw star
                p = (pos+1)*hdim
                for i in range(2):
                    ss = np.random.uniform(-1.0, 1.0, 2) * ssz/2
                    draw.line((p[0]+ss[0], p[1]+ss[1], p[0]-ss[0], p[1]-ss[1]), fill='white', width=1)
                # Raycasts to nearest edges
                t = auv[0] * hdim
                draw.line((hdim, hdim, hdim+t, hdim), fill='red', width=1)
                t = auv[1] * hdim
                draw.line((hdim, hdim, hdim, hdim+t), fill='red', width=1)
                # Full wrap bounds (not directly shown to agent)
                corners = [[1,1],[1,-1],[-1,-1],[-1,1]]
                corners = [(rotate_pt(np.array(c)*WRAP_BOUND-ego.pos, 
                            -ego.ang)+1) * hdim for c in corners]
                prev = corners[-1]
                for c in corners:
                    draw.line([prev[0],prev[1],c[0],c[1]], fill='grey',width=1)
                    prev = c
                # Ship should be drawn in the center 
                pos = np.array([0,0])
        
        p = (pos+1) * hdim
        ppts = np.array([[psz/2, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
        a = ang*np.pi/180
        rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
        ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
        draw.ellipse((p[0]-psz/2, p[1]-psz/2, p[0]+psz/2, p[1]+psz/2), outline='gray' if not terminated else 'lime' if reward>0 else 'red' if reward < 0 else 'yellow')
        draw.polygon(ppts, fill='white')
        # Draw the thruster flare
        if (self.last_act[0]==1):
            ppts = np.array([[-psz*3/4, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
            a = ang*np.pi/180
            rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
            ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
            draw.polygon(ppts, fill='orange')
        # Draw the player's ammo counter
        bar_len = psz * 2
        x_offset = p[0] - bar_len / 2
        y = p[1] + psz + 10
        x = x_offset
        draw.line([(x, y), (x+bar_len, y)], width=1, fill='gray')
        draw.line([(x, y), (x+bar_len*self.stored_missiles/NUM_MISSILES, y)], width=1, fill='white')
        y+=2
        draw.line([(x, y), (x+bar_len*max(0,self.reloadTime)/MISSILE_RELOAD_TIME, y)], width=1, fill='yellow')
        if (ego is not None):
            # Draw an orange line indicating angle unit vector 
            if (ego is not self):
                auv = auv*psz
                draw.line([p[0],p[1], p[0]+auv[0],p[1]+auv[1]], width=1, fill='orange')
            # Draw a cyan line indicating velocity
            vel = vel * psz*dim*2
            draw.line([p[0],p[1], p[0]+vel[0],p[1]+vel[1]], width=1, fill='cyan')