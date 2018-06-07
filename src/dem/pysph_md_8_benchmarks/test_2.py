"""normal impact of a sphere with a rigid wall made of al.alloy

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
    HertzSpringForceParticleWall,
    BodyForce, )
from pysph.solver.application import Application


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
        self.rho = 4000
        self.r = 0.0025
        _m = np.pi * 2 * self.r * 2 * self.r * self.rho
        self.m_eff = (_m + _m) / (_m * _m)

    def create_particles(self):
        # al alloy particles
        x = np.array([-0.10002])
        y = np.array([0])
        r = 0.1
        R = np.ones_like(x) * r
        _m = 4. / 3. * np.pi * r * r * r * 2699
        m = np.ones_like(x) * _m
        E = np.ones_like(x) * 7 * 1e10
        nu = np.ones_like(x) * 0.3
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2. / 5. * _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * 100

        # velocity assigning
        u = np.asarray([0.2])
        v = np.ones_like(x) * 0

        al = get_particle_array_dem(x=x, y=y, u=u, v=v, m=m, E=E, nu=nu,
                                    m_inverse=m_inverse, R=R, h=h,
                                    I_inverse=I_inverse, name="al")

        x = np.array([0])
        y = np.array([0.1])
        E = np.ones_like(x) * 7 * 1e10
        nu = np.ones_like(x) * 0.3
        h = np.ones_like(x) * 100
        nx = np.asarray([-1])

        al_wall = get_particle_array_dem(x=x, y=y, E=E, nu=nu, nx=nx,
                                         name="al_wall")
        # mg alloy particles
        x = np.array([-0.10002])
        y = np.array([1])
        r = 0.1
        R = np.ones_like(x) * r
        _m = 4. / 3. * np.pi * r * r * r * 1800
        m = np.ones_like(x) * _m
        E = np.ones_like(x) * 4 * 1e10
        nu = np.ones_like(x) * 0.35
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2. / 5. * _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * 100

        # velocity assigning
        u = np.asarray([0.2])
        v = np.ones_like(x) * 0

        mg = get_particle_array_dem(x=x, y=y, u=u, v=v, m=m, E=E, nu=nu,
                                    m_inverse=m_inverse, R=R, h=h,
                                    I_inverse=I_inverse, name="mg")

        x = np.array([0])
        y = np.array([1.1])
        E = np.ones_like(x) * 4 * 1e10
        nu = np.ones_like(x) * 0.35
        h = np.ones_like(x) * 100
        nx = np.asarray([-1])

        mg_wall = get_particle_array_dem(x=x, y=y, E=E, nu=nu, nx=nx,
                                         name="mg_wall")
        # additional properties for equations
        # self.m_eff = (_m + mw[0]) / (_m * mw[0])
        return [al, al_wall, mg, mg_wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(mg=DEMStep(), al=DEMStep())

        dt = 1e-6
        print("DT: %s" % dt)
        tf = 0.0013
        solver = Solver(kernel=kernel, dim=3, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                HertzSpringForceParticleWall(dest='al', sources=['al_wall'],
                                             ln_e=abs(np.log(1)), mu=.0),
                HertzSpringForceParticleWall(dest='mg', sources=['mg_wall'],
                                             ln_e=abs(np.log(1)), mu=.0),
                MakeForcesZero(dest='al', sources=None),
                MakeForcesZero(dest='mg', sources=None)
            ]),
        ]
        return equations

    def post_processing(self):
        files = get_files('test_2_output')
        t = []
        y1 = []
        y2 = []
        for solver_data, mg, al in iter_output(files, 'mg', 'al'):
            t.append(solver_data['t'])
            y1.append(-mg.fx[0])
            y2.append(-al.fx[0])
        print("Done")
        plt.plot(t, y1)
        plt.plot(t, y2)
        plt.title("Force in overlap")
        plt.xlabel("time")
        plt.ylabel("Force")
        plt.show()

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
