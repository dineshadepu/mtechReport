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
tf = 0.5
gz = -9.81

hdx = 1.0


def create_ball(x_c, y_c, radius, points):
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
    print(indices)

    # delete the indices outside circle
    x = np.delete(x, indices)
    y = np.delete(y, indices)

    # assign radius for each particle in body
    # rad = np.ones_like(x) * _rad

    return x, y, _rad


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class BallBouncing(Application):
    def initialize(self):
        self.kn = 1e6
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
        xb, yb, _rad = create_ball(0., 0.5 + 0.1 / 2, 0.1 / 2., 30)
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

        xt, yt, _rad = create_ball(0., 0.1 / 2., 0.1 / 2., 30)
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
