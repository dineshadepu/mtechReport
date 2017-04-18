"""Freely falling solid sphere onto a floor under gravity using multi
sphere approach.

This is used to test the rigid body equations and the support for
multiple bodies.

"""
import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)
dim = 2
tf = 1
gz = -9.8

hdx = 1.0


def create_rectangle(xc, yc, length, height, points):
    x = np.linspace(-length / 2., length / 2., points)
    y = np.linspace(-height / 2., height / 2., points)

    # Find radius of single sphere in discretized body
    _rad = (x[2] - x[1]) / 2.

    # get the grid
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()

    # move to to respective position in space
    x = x + xc
    y = y + yc

    return x, y, _rad


def create_floor():
    y = np.linspace(-0.1, -0.5, 3)
    x = np.linspace(0, 300*1e-2, 16)

    x, y = np.meshgrid(x, y)

    x, y = x.ravel(), y.ravel()
    _rad = 10 * 1e-2
    return x, y, _rad


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class BallBouncing(Application):
    def initialize(self):
        self.kn = 1e7
        self.en = 0.9
        self.rad = 10 * 1e-2
        self.rho = 2.7 * 1e3

        self._m = np.pi * self.rad**2 * self.rho

        m_eff = self._m / 2.0
        self.gamma_n = 2 * np.sqrt(m_eff * self.kn) * abs(np.log(self.en)) / (
            np.sqrt(np.pi**2 + np.log(self.en)**2))
        t_c = np.pi * (self.kn / m_eff - self.gamma_n**2 /
                       (4 * m_eff**2))**(-0.5)
        self.dt = t_c / t_c * 1e-4

    def create_particles(self):
        xb, yb, _rad = create_rectangle(150 * 1e-2, 150 * 1e-2, 200 * 1e-2, 200
                                        * 1e-2, 10)
        _m = np.pi * _rad**2 * self.rho
        m = np.ones_like(xb) * _m
        h = np.ones_like(xb) * hdx * self.rad
        ball = get_particle_array_rigid_body(
            name='ball',
            x=xb,
            y=yb,
            h=h,
            m=m, )

        add_properties(ball, 'rad_s')
        # ball.rad_s[:] = 0.05 * 1e-2
        ball.rad_s[:] = _rad
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

        xt, yt, _rad = create_floor()
        _m = np.pi * _rad**2 * self.rho
        m = np.ones_like(xt) * _m
        h = np.ones_like(xt) * hdx * self.rad
        wall = get_particle_array_rigid_body(
            name='wall',
            x=xt,
            y=yt,
            h=h,
            m=m, )

        add_properties(wall, 'rad_s')
        wall.rad_s[:] = _rad

        return [ball, wall]

    def create_solver(self):
        print(self.gamma_n)
        print(self.dt)
        kernel = CubicSpline(dim=dim)

        integrator = EPECIntegrator(ball=RK2StepRigidBody())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                        dt=self.dt, tf=tf, adaptive_timestep=False)
        solver.set_print_freq(10)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='ball', sources=None, gy=gz),
                RigidBodyCollision(
                    dest='ball',
                    sources=['wall'],
                    kn=self.kn,
                    gamma_n=self.gamma_n, )
            ]),
            Group(equations=[RigidBodyMoments(dest='ball', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='ball', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = BallBouncing()
    app.run()
