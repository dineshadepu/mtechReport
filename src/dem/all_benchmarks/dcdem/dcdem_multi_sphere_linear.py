"""Freely falling solid sphere onto a floor under gravity using multi
sphere approach.

This is used to test the rigid body equations and the support for
multiple bodies.

"""
import numpy as np
import matplotlib.pyplot as plt

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import (BodyForce, RigidBodyWallCollision,
                                  RigidBodyCollision,
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)
dim = 2
wall_time = 0.1
tf = wall_time + 0.5
gz = -9.81
points = 20

hdx = 1.0


def create_ball(x_c, y_c, radius, points, rad=True):
    """Given radius of big sphere, this function discretizes it into many
small spheres, Returns x, y numpy arrays, with corresponding radius

    """
    x = np.linspace(x_c - radius, x_c + radius, points)
    y = np.linspace(y_c - radius, y_c + radius, points)

    # Find radius of single sphere in discretized body
    _dia = x[2] - x[1]
    _rad = _dia / 2

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
    xb = np.linspace(130, 260, 1)
    yb = np.ones_like(xb) * 0
    nxb = np.ones_like(xb) * 0
    nyb = np.ones_like(xb) * 1

    yl = np.arange(0, 260, 10)
    xl = np.ones_like(yl) * 0
    nxl = np.ones_like(xl) * 1
    nyl = np.ones_like(xl) * 0

    yr = np.arange(0, 260, 10)
    xr = np.ones_like(yl) * 261
    nxr = np.ones_like(xr) * -1
    nyr = np.ones_like(xr) * 0

    x = np.concatenate([xl, xb, xr])
    y = np.concatenate([yl, yb, yr])
    nx = np.concatenate([nxl, nxb, nxr])
    ny = np.concatenate([nyl, nyb, nyr])
    return x * 1e-3, y * 1e-3, nx, ny


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


def create_temp_wall():
    y_b = np.arange(0, 70, 10)
    x_b = np.ones_like(y_b) * 60
    nx = np.ones_like(x_b) * -1
    ny = np.ones_like(x_b) * 0
    return x_b * 1e-3, y_b * 1e-3, nx, ny


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

        self._m = np.pi * self.rad**2 * self.rho

        m_eff = self._m / 2.0
        self.gamma_n = 2 * np.sqrt(m_eff * self.kn) * abs(np.log(self.en)) / (
            np.sqrt(np.pi**2 + np.log(self.en)**2))
        t_c = np.pi * (self.kn / m_eff - self.gamma_n**2 /
                       (4 * m_eff**2))**(-0.5)
        self.dt = t_c / t_c * 1e-4

    def create_particles(self):
        xb, yb, body_id = create_six_layers()
        yb = yb - 0.9*1e-2

        x, y, _rad = create_ball(0.5 * 1e-2, 1.6 * 1e-2, 0.5 * 1e-2, points)
        _m = np.pi * _rad**2 * self.rho

        m = np.ones_like(xb) * _m
        h = np.ones_like(xb) * hdx * self.rad
        ball = get_particle_array_rigid_body(
            name='ball',
            x=xb,
            y=yb,
            h=h,
            m=m,
            body_id=body_id)

        add_properties(ball, 'rad_s')
        add_properties(ball, 'youngs')
        add_properties(ball, 'poisson')
        ball.rad_s[:] = _rad
        ball.youngs[:] = 69 * 1e9
        ball.poisson[:] = 0.30
        add_properties(
            ball,
            'tang_disp_x',
            'tang_disp_y',
            'tang_disp_z',
            'tang_disp_x0',
            'tang_disp_y0',
            'tang_disp_z0',
            'tang_velocity_x',
            'tang_velocity_y',
            'tang_velocity_z', )

        xt, yt, nx, ny = create_boundary()
        # dx is manually given
        h = np.ones_like(xt) * hdx * 130 * 1e-3
        wall = get_particle_array_rigid_body(
            name='wall',
            x=xt,
            y=yt,
            nx=nx,
            ny=ny,
            h=h,
            )

        xtw, ytw, nx, ny = create_temp_wall()
        h = np.ones_like(xtw) * hdx * self.rad
        temp_wall = get_particle_array_rigid_body(
            name='temp_wall',
            x=xtw,
            y=ytw,
            nx=nx,
            ny=ny,
            h=h,
            )
        return [ball, wall, temp_wall]

    def create_solver(self):
        print(self.gamma_n)
        print(self.dt)
        kernel = CubicSpline(dim=dim)

        integrator = EPECIntegrator(ball=RK2StepRigidBody())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                        dt=self.dt, tf=tf, output_at_times=[wall_time-0.025,
                                                            wall_time-0.05,
                                                            wall_time+0.1,
                                                            wall_time+0.3,
                                                            wall_time+0.5],
                        pfreq=1000000)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='ball', sources=None, gy=gz),
                RigidBodyCollision(
                    dest='ball',
                    sources=['ball'],
                    kn=1e5,
                    en=0.1,
                    mu=0.25),
                RigidBodyWallCollision(
                    dest='ball',
                    sources=['wall', 'temp_wall'],
                    kn=1e5,
                    en=0.2,
                    mu=0.25)
            ]),
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


if __name__ == '__main__':
    app = BallBouncing()
    app.run()
    # x, y, _rad = create_ball(3*1e-2, 5*1e-2, 0.5*1e-2, 20)
    # x, y, body_id = create_six_layers()
    # xtw, ytw = create_temp_wall()
    # plt.scatter(x, y)
    # plt.scatter(xw, yw)
    # plt.scatter(xtw, ytw)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()
