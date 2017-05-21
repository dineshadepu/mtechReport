"""Oblique impact of a sphere with a rigid plane with a
   constant resultant velocity but at different incident angle
   Do pfreq = 1000

Check the complete molecular dynamics code
"""
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_dem
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.utils import get_files, iter_output
from pysph.sph.integrator_step import DEMStep

from pysph.sph.equation import Group
from pysph.sph.molecular_dynamics import (LinearSpringForceParticleParticle,
                                          HertzSpringForceParticleWall,
                                          MakeForcesZero, BodyForce)
from pysph.solver.application import Application

# thet = sys.argv[1]
thet = 80


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
        _m = 4. / 3. * np.pi * self.r * self.r * self.r * self.rho
        self.m_eff = (_m + _m) / (_m * _m)

    def create_particles(self):
        x = np.array([0.0])
        y = np.array([0.003])
        x = np.array([0.0])
        r = 0.0025
        R = np.ones_like(x) * r
        _m = 4. / 3. * np.pi * self.r * self.r * self.r * self.rho
        m = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2. / 5. * _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * 10
        E = np.ones_like(x) * 3.8 * 1e11
        nu = np.ones_like(x) * 0.23

        # velocity assigning
        Vmag = 3.9
        # thet = 40
        theta = thet * np.pi / 180.
        u = Vmag * np.sin(theta) * np.ones_like(x)
        v = -Vmag * np.cos(theta) * np.ones_like(x)

        sand = get_particle_array_dem(x=x, y=y, u=u, v=v, m=m, E=E, nu=nu,
                                      m_inverse=m_inverse, R=R, h=h,
                                      I_inverse=I_inverse, name="sand")

        x, y = np.asarray([0]), np.asarray([0])
        nx, ny = np.asarray([0]), np.asarray([1])
        R = np.ones_like(x) * r
        h = np.ones_like(x) * 10
        wall = get_particle_array_dem(x=x, y=y, E=E, nu=nu,
                                      nx=nx, ny=ny,
                                      h=h,  name="wall")

        # additional properties for equations
        # self.m_eff = (_m + mw[0]) / (_m * mw[0])
        return [sand, wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(sand=DEMStep())

        dt = 1e-7
        print("DT: %s" % dt)
        tf = 0.0015
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False, pfreq=2000)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='sand', sources=None, gy=-9.81),
                HertzSpringForceParticleWall(
                    dest='sand', sources=['wall'],
                    ln_e=abs(np.log(0.98)),  mu=.092),
                # LinearSpringForceParticleParticle(
                #     dest='sand', sources=['wall'],
                #     ln_e=abs(np.log(0.98)), m_eff=self.m_eff, mu=.092),
                # MakeForcesZero(dest='sand', sources=None)
            ]),
        ]
        return equations

    def post_processing(self):
        files = get_files('test_4_output')
        t = []
        y1 = []
        for solver_data, sand, wall in iter_output(files, 'sand', 'wall'):
            t.append(solver_data['t'])
            y1.append(sand.wz[0])
        print("Done")
        target = open('output.txt', 'a+')
        target.write(str(y1[-1]))
        target.close()
        print(y1[-1])
    # def pre_step(self, solver):
    #     solver.dump_output()


if __name__ == '__main__':
    app = FluidStructureInteration()
    app.run()
    app.post_processing()
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
