import numpy as np

# constants
MISSILE_VEL = 0.005
MISSILE_LIFE = 108
SHIP_TURN_RATE = 1.5
GRAV_CONST = .00000125

STAR_SIZE = .01 # Star size
PLAYER_SIZE = .02 # Player size
WRAP_BOUND = .5**.5 # x/y distance from star at which wrapping occurs

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
    wrap(diff) # Wrap adjusted position by wrapping dist
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
        wrap(mp)
        self.life -= 1 * speed
        return -1, (self.life<=0)
    def get_obs(self, ego):
        if (ego is None):
            return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.life/self.maxLife], dtype=np.float32)
        else:
            p = ego_pt(self.pos, ego)
            v = rotate_pt(self.vel, -ego.ang) # Rotate velocity w/r to removing ego's angle
            obs = np.concatenate([p, v, [self.life/self.maxLife]], dtype=np.float32)
            return obs

class Ship():
    REPR_SIZE = 7
    def __init__(self, pos, ang, size=PLAYER_SIZE, vel=None):
        self.pos = pos
        self.ang = ang
        self.stored_missiles = 1
        self.vel = vel if vel is not None else np.array([0.,0.])
        self.size = size
        self.updateAngUV()
    def updateAngUV(self):
        a = self.ang*np.pi/180
        self.angUV = np.array([np.cos(a), -np.sin(a)])
    def update(self, action, missiles, speed):
        # Take actions
        if (action[0]==1):
          self.ang += SHIP_TURN_RATE * speed
          self.updateAngUV()
        elif (action[0]==2):
          self.ang -= SHIP_TURN_RATE * speed
          self.updateAngUV()
        # Shoot
        if (action[1]==1 and self.stored_missiles > 0):
            m = Missile(self.pos + self.angUV * self.size,
                self.vel + self.angUV * MISSILE_VEL)
            missiles.append(m)
            self.stored_missiles -= 1
    def get_obs(self,
            ego=None, # The Ship associated with the observing agent
            ):
        # Self: [star, vel, to_edges, missiles]
        # Other: [pos (relative to self), vel (rotated to self), auv (rotated to self), missiles, reload time]
        if (ego==self):
            p = np.array([0,0]) # We do not move in simplified env
            auv = get_raycasts(self.pos, self.ang) # Raycasts to arena boundary
        else:
            p = ego_pt(self.pos, ego)
            auv = np.array([0,0]) # Angle of other ship is not relevant in simplified environment
        v = rotate_pt(self.vel, -ego.ang) # Rotate velocity w/r to removing ego's angle
        obs = np.concatenate([p, v, auv,
            [self.stored_missiles]
        ], dtype=np.float32)
        return obs.clip(-1,1) # avoid rounding errors

class Dummy_Ship(Ship):
    def update(self, speed):
        self.updateAngUV()
        # Update position
        self.pos += self.vel * speed
        self.vel = np.clip(self.vel, -1.0, 1.0)
        wrap(self.pos)
        # Apply force of gravity. GMm can be treated as a single constant.
        self.vel -= (self.pos * GRAV_CONST / (self.pos[0]**2 + self.pos[1]**2)** 1.5) * speed