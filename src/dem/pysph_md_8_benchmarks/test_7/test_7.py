"""normal impact of a sphere with a other sphere made of glass,
remember for preq=5 or 1

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
from pysph.solver.utils import get_files, iter_output
from pysph.sph.integrator_step import DEMStep

from pysph.sph.equation import Group
from pysph.sph.molecular_dynamics import (
    MakeForcesZero,
    HertzSpringForceParticleParticle,
    BodyForce, )
from pysph.solver.application import Application

omega = 10


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


def create_wall(r):
    x = np.arange(-0.04, 0.06, 2 * r)
    y = np.array([0])
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()
    return x, y


class FluidStructureInteration(Application):
    def initialize(self):
        self.dx = 1

    def create_particles(self):
        # al particles
        x_pos = 0.102
        x = np.array([-x_pos, 0.10])
        y = np.array([0, 0])
        r = 0.1
        R = np.ones_like(x) * r
        _m = 4. / 3. * np.pi * r * r * r * 2700
        m = np.ones_like(x) * _m
        E = np.ones_like(x) * 7 * 1e10
        nu = np.ones_like(x) * 0.33
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2./5. * _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * r

        # velocity assigning
        u = np.asarray([0.2, -0.2])
        v = np.ones_like(x) * 0
        wz = np.asarray([omega, -omega])

        al = get_particle_array_dem(x=x, y=y, u=u, v=v, wz=wz, m=m, E=E,
                                    nu=nu,
                                    m_inverse=m_inverse, R=R, h=h,
                                    I_inverse=I_inverse, name="al")

        # copper particles
        x = np.array([-x_pos, 0.10])
        y = np.array([0.5, 0.5])
        r = 0.1
        R = np.ones_like(x) * r
        _m = 4. / 3. * np.pi * r * r * r * 8900
        m = np.ones_like(x) * _m
        E = np.ones_like(x) * 1.2 * 1e11
        nu = np.ones_like(x) * 0.35
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2./5. * _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * r

        # velocity assigning
        u = np.asarray([0.2, -.2])
        v = np.ones_like(x) * 0
        wz = np.asarray([omega, -omega])

        copper = get_particle_array_dem(x=x, y=y, u=u, v=v, wz=wz, m=m, E=E,
                                        nu=nu,
                                        m_inverse=m_inverse, R=R, h=h,
                                        I_inverse=I_inverse, name="copper")
        # additional properties for equations
        # self.m_eff = (_m + mw[0]) / (_m * mw[0])
        return [al, copper]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(al=DEMStep(), copper=DEMStep())

        dt = 1e-6
        print("DT: %s" % dt)
        tf = 0.007
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False, pfreq=1000)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                HertzSpringForceParticleParticle(
                    dest='al', sources=['al'],
                    ln_e=abs(np.log(0.5)), mu=.40),
                HertzSpringForceParticleParticle(
                    dest='copper', sources=['copper'],
                    ln_e=abs(np.log(0.5)), mu=.40),
                MakeForcesZero(dest='al', sources=None),
                MakeForcesZero(dest='copper', sources=None)
            ]),
        ]
        return equations

    def post_processing(self):
        files = get_files('test_7_output')
        t = []
        y1 = []
        y2 = []
        for solver_data, al, copper in iter_output(files, 'al', 'copper'):
            t.append(solver_data['t'])
            y1.append(-al.wz[0])
            y2.append(-copper.wz[0])
        print("Done")
        print(y1)
        print(y2)
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
