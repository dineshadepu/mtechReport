"""Freely falling stack of spheres into vessel of water a floor under
gravity using multi sphere approach.


"""
import numpy as np
import matplotlib.pyplot as plt

from pysph.base.kernels import CubicSpline
from pysph.base.utils import (get_particle_array_rigid_body,
                              get_particle_array_wcsph)
from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation,)
from pysph.sph.wc.basic import TaitEOS, MomentumEquation

from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
                                  RigidBodyMoments, RigidBodyMotion,
                                  SolidFluidForce,
                                  RK2StepRigidBody)
dim = 2
tf = 1.
wall_time = 0.5
gz = -9.81
points = 20

hdx = 1.2


def create_ball(x_c, y_c, radius, points, rad=True):
    """Given radius of big sphere, this function discretizes it into many
small spheres, Returns x, y numpy arrays, with corresponding radius

    """
    x = np.linspace(x_c - radius, x_c + radius, points)
    y = np.linspace(y_c - radius, y_c + radius, points)

    # Find radius of single sphere in discretized body
    _rad1 = x[2] - x[1]
    _rad = _rad1 / 2

    # get the grid
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()

    # get the indices outside circle
    indices = []
    for i in range(len(x)):
        if (x_c - x[i])**2 + (y_c - y[i])**2 >= radius**2:
            indices.append(i)

    # delete the indices outside circle
    x = np.delete(x, indices)
    y = np.delete(y, indices)

    # assign radius for each particle in body
    # rad = np.ones_like(x) * _rad

    if rad is True:
        return x, y, _rad
    else:
        return x, y


def create_boundary():
    # every thing here is in mm milli metre
    # spacing in x direction is taken as 1 mm
    dx = 1
    x_b = np.arange(0, 260, dx)
    y_b = np.arange(0, -2*dx, -dx)
    x_b, y_b = np.meshgrid(x_b, y_b)
    x_b, y_b = x_b.ravel(), y_b.ravel()

    x_l = np.arange(-2 * dx, 0, dx)
    y_l = np.arange(-dx, 200, dx)
    x_l, y_l = np.meshgrid(x_l, y_l)
    x_l, y_l = x_l.ravel(), y_l.ravel()

    x_r = np.arange(260, 260 + 2*dx, dx)
    y_r = y_l
    x_r, y_r = np.meshgrid(x_r, y_r)
    x_r, y_r = x_r.ravel(), y_r.ravel()

    x = np.concatenate([x_l, x_b, x_r])
    y = np.concatenate([y_l, y_b, y_r])

    return x * 1e-3, y * 1e-3


def create_six_layers():
    x1, y1 = create_ball(0.5 * 1e-2, 1.6 * 1e-2, 0.5 * 1e-2, points, rad=False)
    # print(len(x1))
    x2, y2 = x1 + 1 * 1e-2, y1
    x3, y3 = x2 + 1 * 1e-2, y1
    x4, y4 = x3 + 1 * 1e-2, y1
    x5, y5 = x4 + 1 * 1e-2, y1
    x6, y6 = x5 + 1 * 1e-2, y1

    # Bottom first layer is done
    x_six_bot, y_six_bot = np.concatenate(
        [x1, x2, x3, x4, x5, x6]), np.concatenate([y1, y2, y3, y4, y5, y6])

    x, y = x_six_bot, y_six_bot

    # middle six cylinders
    y_middle = y_six_bot + 2.2 * 1e-2
    x, y = np.concatenate([x, x_six_bot]), np.concatenate([y, y_middle])

    # top six cylinders
    y_top = y_middle + 2.2 * 1e-2
    x, y = np.concatenate([x, x_six_bot]), np.concatenate([y, y_top])

    # Bottom first 5 cylinder layer
    x1, y1 = create_ball(1 * 1e-2, 2.7 * 1e-2, 0.5 * 1e-2, points, rad=False)
    x2, y2 = x1 + 1 * 1e-2, y1
    x3, y3 = x2 + 1 * 1e-2, y1
    x4, y4 = x3 + 1 * 1e-2, y1
    x5, y5 = x4 + 1 * 1e-2, y1

    y_five_bottom = np.concatenate([y1, y2, y3, y4, y5])
    x_five_bottom = np.concatenate([x1, x2, x3, x4, x5])

    x, y = np.concatenate([x, x_five_bottom]), np.concatenate(
        [y, y_five_bottom])

    # middle six cylinders
    y_middle = y_five_bottom + 2.2 * 1e-2
    x, y = np.concatenate([x, x_five_bottom]), np.concatenate([y, y_middle])

    # top six cylinders
    y_top = y_middle + 2.2 * 1e-2
    x, y = np.concatenate([x, x_five_bottom]), np.concatenate([y, y_top])

    # create body id
    _b_id = np.ones_like(x1, dtype=int)
    body_id = np.asarray([], dtype=int)
    for i in range(33):
        body_id = np.append(body_id, i * _b_id)
    return x, y, body_id


def create_temp_wall(dx=1):
    right_end = 100
    x_b = np.arange(right_end, right_end + 2*dx, dx)
    y_b = np.arange(dx, 120, dx)
    x_b, y_b = np.meshgrid(x_b, y_b)
    x_b, y_b = x_b.ravel(), y_b.ravel()

    return x_b * 1e-3, y_b * 1e-3


def create_fluid(dx=1):
    xf = np.arange(0, 100, dx)
    yf = np.arange(dx, 80, dx)

    xf, yf = np.meshgrid(xf, yf)
    xf = xf.ravel()
    yf = yf.ravel()
    return xf*1e-3, yf*1e-3


def single_layer():
    x1, y1 = create_ball(0.5 * 1e-2, 1.6 * 1e-2, 0.5 * 1e-2, points, rad=False)
    # print(len(x1))
    x2, y2 = x1 + 1.5 * 1e-2, y1
    x3, y3 = x2 + 1.5 * 1e-2, y1
    x4, y4 = x3 + 1.5 * 1e-2, y1
    x5, y5 = x4 + 1.5 * 1e-2, y1
    x6, y6 = x5 + 1.5 * 1e-2, y1

    # Bottom first layer is done
    x_six_bot, y_six_bot = np.concatenate(
        [x1, x2, x3, x4, x5, x6]), np.concatenate([y1, y2, y3, y4, y5, y6])

    x, y = x_six_bot, y_six_bot

    # scale y top of fluid
    y = y + 0.08

    # create body id
    _b_id = np.ones_like(x1, dtype=int)
    body_id = np.asarray([], dtype=int)

    for i in range(6):
        body_id = np.append(body_id, i * _b_id)

    return x, y, body_id


def two_spheres():
    x1, y1 = create_ball(0.5 * 1e-2, 1.6 * 1e-2, 0.5 * 1e-2, points, rad=False)
    # print(len(x1))
    # move x1 to little right
    x1 = x1 + 0.01
    x2, y2 = x1 + 4.5 * 1e-2, y1 + 0.02
    # Bottom first layer is done
    x_six_bot, y_six_bot = np.concatenate(
        [x1, x2]), np.concatenate([y1, y2])

    x, y = x_six_bot, y_six_bot

    # scale y top of fluid
    y = y + 0.08

    # create body id
    _b_id = np.ones_like(x1, dtype=int)
    body_id = np.asarray([], dtype=int)

    for i in range(2):
        body_id = np.append(body_id, i * _b_id)

    return x, y, body_id


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class BallBouncing(Application):
    def initialize(self):
        self.en = 0.2
        self.rad = 0.1 * 1e-2
        A = np.pi * self.rad**2
        self.kn = 69 * 1e9 * A / (2 * self.rad)
        print(self.kn)
        self.rho = 2.7 * 1e3

        self.ro = 1000
        self.co = 2 * np.sqrt(2 * 9.81 * 60 * 1e-3)
        self.alpha = 0.1

        self._m = np.pi * self.rad**2 * self.rho

        m_eff = self._m / 2.0
        self.gamma_n = 2 * np.sqrt(m_eff * self.kn) * abs(np.log(self.en)) / (
            np.sqrt(np.pi**2 + np.log(self.en)**2))
        t_c = np.pi * (self.kn / m_eff - self.gamma_n**2 /
                       (4 * m_eff**2))**(-0.5)
        self.dt = 4 * t_c / t_c * 1e-5

    def create_particles(self):
        # xb, yb, body_id = single_layer()
        # xb, yb, body_id = two_spheres()
        # xb, yb, body_id = create_six_layers()
        xb, yb, _rad = create_ball(0.5 * 1e-2, 1.6 * 1e-2, 0.5 * 1e-2, points)
        yb = yb + 0.08
        body_id = np.zeros_like(xb, dtype=int)

        x, y, _rad = create_ball(0.5 * 1e-2, 1.6 * 1e-2, 0.5 * 1e-2, points)
        _m = np.pi * _rad * _rad * 2120

        m = np.ones_like(xb) * _m
        h = np.ones_like(xb) * hdx * self.rad * 2
        rho = np.ones_like(xb) * 2120
        rad_s = np.ones_like(xb) * _rad
        ball = get_particle_array_rigid_body(
            name='ball',
            x=xb,
            y=yb,
            h=h,
            rho=rho,
            m=m,
            rad_s=rad_s,
            body_id=body_id)

        dx = 1 * 1e-3
        xt, yt = create_boundary()
        _m = np.pi * 1000 * dx * dx
        m = np.ones_like(xt) * _m
        rho = np.ones_like(xt) * 1000
        h = np.ones_like(xt) * hdx * dx
        wall = get_particle_array_wcsph(
            name='wall',
            x=xt,
            y=yt,
            h=h,
            rho=rho,
            m=m, )

        add_properties(wall, 'rad_s')
        wall.rad_s[:] = (dx / 2.)

        xtw, ytw = create_temp_wall()
        m = np.ones_like(xtw) * _m
        h = np.ones_like(xtw) * hdx * dx
        rho = np.ones_like(xtw) * 1000
        temp_wall = get_particle_array_wcsph(
            name='temp_wall',
            x=xtw,
            y=ytw,
            rho=rho,
            h=h,
            m=m, )

        add_properties(temp_wall, 'rad_s')
        temp_wall.rad_s[:] = (dx / 2.)

        xf, yf = create_fluid()
        rho = np.ones_like(xf) * 1000
        m = rho[:] * dx * dx
        h = np.ones_like(xf) * hdx * dx
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho,
                                         name="fluid")

        return [ball, wall, temp_wall, fluid]

    def create_solver(self):
        print(self.gamma_n)
        print(self.dt)
        kernel = CubicSpline(dim=dim)

        # integrator = EPECIntegrator(ball=RK2StepRigidBody())
        integrator = EPECIntegrator(wall=WCSPHStep(), temp_wall=WCSPHStep(),
                                    fluid=WCSPHStep(), ball=RK2StepRigidBody())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                        dt=self.dt, tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='ball', sources=None, gy=gz),
            ]),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=self.ro, c0=self.co,
                        gamma=7.0),
                TaitEOS(dest='wall', sources=None, rho0=self.ro, c0=self.co,
                        gamma=7.0),
                TaitEOS(dest='temp_wall', sources=None, rho0=self.ro,
                        c0=self.co, gamma=7.0),
            ], real=False),

            Group(equations=[
                ContinuityEquation(
                    dest='fluid',
                    sources=['fluid', 'temp_wall', 'wall'],),
                ContinuityEquation(
                    dest='temp_wall',
                    sources=['fluid', 'temp_wall', 'wall'],),
                ContinuityEquation(
                    dest='wall',
                    sources=['fluid', 'temp_wall', 'wall'],),
                MomentumEquation(dest='fluid', sources=['fluid', 'wall',
                                                        'temp_wall'],
                                 alpha=self.alpha, beta=0.0, c0=self.co,
                                 gy=-9.81),
                SolidFluidForce(dest='fluid', sources=['ball'],),

                XSPHCorrection(dest='fluid', sources=['fluid', 'temp_wall',
                                                      'wall']),
            ]),
            Group(equations=[RigidBodyCollision(dest='ball',
                                                sources=['ball', 'wall',
                                                         'temp_wall'],
                                                kn=1e5)]),
            Group(equations=[RigidBodyMoments(dest='ball', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='ball', sources=None)]),
        ]
        return equations

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        T = wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'temp_wall':
                    break
            pa.y += 8 * 1e-2

    def plot_pars(self):
        # x, y, b = create_six_layers()
        # x, y = create_ball(0.5 * 1e-2, 1.6 * 1e-2, 0.5 * 1e-2, points,
        #                    rad=False)
        # x, y, b_id = single_layer()
        x, y, b_id = two_spheres()
        xw, yw = create_boundary()
        xtw, ytw = create_temp_wall()
        xf, yf = create_fluid()
        plt.scatter(x, y)
        plt.scatter(xw, yw)
        plt.scatter(xtw, ytw)
        plt.scatter(xf, yf)
        plt.axes().set_aspect('equal', 'datalim')
        print("plotted")
        plt.show()


if __name__ == '__main__':
    app = BallBouncing()
    # app.plot_pars()
    app.run()

#  LocalWords:  milli
