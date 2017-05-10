"""Oblique impact of a sphere with a rigid plane with a
   constant resultant velocity but at different incident angle

Check the complete molecular dynamics code
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_dem
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import DEMStep

from pysph.sph.equation import Group
from pysph.sph.molecular_dynamics import (LinearSpringForceParticleParticle,
                                          MakeForcesZero, BodyForce)
from pysph.solver.application import Application


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


def create_wall(r):
    x = np.arange(-0.04, 0.06, 2*r)
    y = np.array([0])
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()
    return x, y


class FluidStructureInteration(Application):
    def initialize(self):
        self.dx = 1
        self.rho = 4000
        self.r = 0.0025
        _m = np.pi * 2 * self.r * 2 * self.r * self.rho
        self.m_eff = (_m + _m) / (_m * _m)

    def create_particles(self):
        x = np.array([0.0])
        y = np.array([0.005])
        r = 0.0025
        R = np.ones_like(x) * r
        _m = np.pi * 2 * r * 2 * r * 4000
        m = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        _I = _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * r

        # velocity assigning
        Vmag = 3.9
        theta = 80 * np.pi / 180.
        u = Vmag * np.sin(theta) * np.ones_like(x)
        v = -Vmag * np.cos(theta) * np.ones_like(x)

        sand = get_particle_array_dem(x=x, y=y, u=u, v=v, m=m,
                                      m_inverse=m_inverse, R=R, h=h,
                                      I_inverse=I_inverse, name="sand")

        x, y = create_wall(r)
        mw = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        R = np.ones_like(x) * r
        h = np.ones_like(x) * r
        _I = _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        wall = get_particle_array_dem(x=x, y=y, m=mw, m_inverse=m_inverse, R=R,
                                      h=h, I_inverse=I_inverse, name="wall")

        # additional properties for equations
        # self.m_eff = (_m + mw[0]) / (_m * mw[0])
        return [sand, wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(sand=DEMStep())

        dt = 1e-6
        print("DT: %s" % dt)
        tf = 0.0076
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='sand', sources=None, gy=-9.81),
                LinearSpringForceParticleParticle(
                    dest='sand', sources=['sand', 'wall'], k=1e6,
                    ln_e=abs(np.log(0.98)), m_eff=self.m_eff, mu=.092),
                # MakeForcesZero(dest='sand', sources=None)
            ]),
        ]
        return equations

    # def pre_step(self, solver):
    #     solver.dump_output()


if __name__ == '__main__':
    app = FluidStructureInteration()
    app.run()
    # x, y = create_hopper(0.1)
    # x, y = create_fluid()
    # xc, yc, indices = create_cube()
    # xt, yt = create_boundary(1 * 1e-3)
    # plt.scatter(x, y)
    # plt.scatter(xc, yc)
    # plt.scatter(xt, yt)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()
    # xt, yt = create_boundary(1 * 1e-3)
    # xc, yc, indices = create_cube()
    # xf, yf = create_fluid_with_solid_cube()
    # plt.scatter(xt, yt)
    # plt.scatter(xc, yc)
    # plt.scatter(xf, yf)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()

#  LocalWords:  SummationDensityShepardFilter
