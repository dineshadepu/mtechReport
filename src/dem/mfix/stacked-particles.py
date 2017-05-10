"""Two Stacked Particles Compressed between Two Boundaries

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
    x = np.arange(-0.04, 0.06, 2 * r)
    y = np.array([0])
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()
    return x, y


class FluidStructureInteration(Application):
    def initialize(self):
        self.dx = 1
        self.rho = 4000
        self.rp = 0.05 * 1e-2

        self.m1 = 4. / 3. * np.pi * self.rp**3 * 20000
        self.m2 = 4. / 3. * np.pi * self.rp**3 * 10000
        self.m_eff = (self.m1 + self.m2) / (self.m1 * self.m2)

    def create_particles(self):
        rp = 0.05 * 1e-2
        y_w = 3.6 * rp
        y_1 = 0.25 * y_w
        y_2 = 0.75 * y_w
        y_sand = np.asarray([y_1, y_2])
        x_sand = np.ones_like(y_sand) * 0
        R = np.ones_like(x_sand) * rp
        _m1 = 4. / 3. * np.pi * rp**3 * 20000
        _m2 = 4. / 3. * np.pi * rp**3 * 10000
        m = np.asarray([_m1, _m2])
        m_inverse = np.asarray([1. / _m1, 1. / _m2])
        _I1 = 2. / 5. * _m1 * rp**2
        _I2 = 2. / 5. * _m2 * rp**2
        I_inverse = np.asarray([1. / _I1, 1. / _I2])
        h = np.ones_like(x_sand) * rp

        sand = get_particle_array_dem(x=x_sand, y=y_sand, m=m,
                                      m_inverse=m_inverse, R=R, h=h,
                                      I_inverse=I_inverse, name="sand")

        x_l_w = np.asarray([0])
        y_l_w = np.asarray([-rp])
        mw = np.ones_like(x_l_w) * _m1
        m_inverse = np.ones_like(x_l_w) * 1. / _m1
        R = np.ones_like(x_l_w) * rp
        h = np.ones_like(x_l_w) * rp
        I_inverse = np.ones_like(x_l_w) * 1. / _I1
        lower_wall = get_particle_array_dem(x=x_l_w, y=y_l_w, m=mw,
                                            m_inverse=m_inverse, R=R, h=h,
                                            I_inverse=I_inverse, name="l_wall")

        x_u_w = np.asarray([0])
        y_u_w = np.asarray([y_w + rp])
        mw = np.ones_like(x_u_w) * _m1
        m_inverse = np.ones_like(x_u_w) * 1. / _m1
        R = np.ones_like(x_u_w) * rp
        h = np.ones_like(x_u_w) * rp
        I_inverse = np.ones_like(x_u_w) * 1. / _I1
        upper_wall = get_particle_array_dem(x=x_u_w, y=y_u_w, m=mw,
                                            m_inverse=m_inverse, R=R, h=h,
                                            I_inverse=I_inverse, name="u_wall")
        # additional properties for equations
        # self.m_eff = (_m + mw[0]) / (_m * mw[0])
        return [sand, lower_wall, upper_wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(sand=DEMStep())

        dt = 1e-5
        print("DT: %s" % dt)
        tf = 0.1
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='sand', sources=None, gy=-9.81),
                LinearSpringForceParticleParticle(
                    dest='sand', sources=['sand'], k=1e2,
                    ln_e=abs(np.log(1)), m_eff=self.m_eff, mu=0),
                LinearSpringForceParticleParticle(
                    dest='sand', sources=['l_wall'], k=1e2,
                    ln_e=abs(np.log(1)), m_eff=self.m1, mu=0),
                LinearSpringForceParticleParticle(
                    dest='sand', sources=['u_wall'], k=1e2,
                    ln_e=abs(np.log(1)), m_eff=self.m2, mu=0),
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
