import numpy as np


class Particle(object):
    """Documentation for Particle

    """

    def __init__(self, R=0, x=0, y=0, ang=0, vx=0, vy=0, angv=0, ax=0, ay=0,
                 anga=0, fx=0, fy=0, t=0, p=0, rho=0, nu=0.2, E=4.8 * 1e10,
                 e=1, mu=0, fixed=False):
        super(Particle, self).__init__()
        self.R = R
        self.m = 4. / 3. * np.pi * rho * R * R * R
        # self.m = np.pi * rho * R * R
        self.I = 0.5 * self.m * R * R
        self.x = x
        self.y = y
        self.ang = ang
        self.vx = vx
        self.vy = vy
        self.angv = angv
        self.ax = ax
        self.ay = ay
        self.anga = anga
        self.fx = fx
        self.fy = fy
        self.t = t
        self.p = p
        self.fixed = fixed
        self.nu = nu
        self.E = E
        self.e = e
        self.mu = mu
        self.tang_disp = 0


class Wall(object):
    """Documentation for Wall

    """

    def __init__(self, pos, E, nu):
        super(Wall, self).__init__()
        self.pos = pos
        self.E = E
        self.nu = nu
