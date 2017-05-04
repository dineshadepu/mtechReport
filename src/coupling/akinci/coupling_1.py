"""Water rested in a vessel

Check basic equations of SPH to throw a ball inside the vessel
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation,
                                       SummationDensity)
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody, FluidForceOnSolid,
                                  SolidForceOnFluid)


def create_fluid_with_solid_cube(dx=2 * 1e-3):
    x = np.arange(0, 150 * 1e-3 + 1e-9, dx)
    y = np.arange(0, 130 * 1e-3 + 1e-9, dx)

    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()

    indices = []
    for i in range(len(x)):
        if 63 * 1e-3 < x[i] < 87 * 1e-3:
            if y[i] >= 120 * 1e-3:
                indices.append(i)

    x, y = np.delete(x, indices), np.delete(y, indices)
    return x, y


def create_fluid():
    x = np.arange(0, 10 + 1e-9, 0.1)
    y = np.arange(0, 8 + 1e-9, 0.1)
    x, y = np.meshgrid(x, y)
    x, y = x.ravel() * 1e-2, y.ravel() * 1e-2
    return x, y


def create_boundary():
    # Bottom of the tank
    x = np.arange(-3 * 0.1, 10.3 + 1e-9, 0.1)
    y = np.arange(-3 * 0.1, 10 + 1e-9, 0.1)

    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()

    indices = []
    for i in range(len(x)):
        if x[i] <= 10 and x[i] >= 0:
            if y[i] >= 0:
                indices.append(i)

    x = np.delete(x, indices)
    y = np.delete(y, indices)
    return x * 1e-2, y * 1e-2


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class FluidStructureInteration(Application):
    def initialize(self):
        self.dx = .1 * 1e-2
        self.hdx = 1.2
        self.ro = 1000
        self.co = 2 * np.sqrt(9.81 * 10 * 1e-2)
        self.alpha = 0.25

    def create_particles(self):
        """Create the circular patch of fluid."""
        # xf, yf = create_fluid_with_solid_cube()
        xf, yf = create_fluid()
        m = 0.1 * 0.1 * 1e-2 * 1e-2 * 1000 * np.ones_like(xf)
        rho = np.ones_like(xf) * 1000
        h = np.ones_like(xf) * self.hdx * 0.1 * 1e-2
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho,
                                         name="fluid")

        xt, yt = create_boundary()
        m = np.ones_like(xt) * 1400 * 0.1 * 0.1 * 1e-2 * 1e-2
        rho = np.ones_like(xt) * 1000
        h = np.ones_like(xt) * self.hdx * 0.1 * 1e-2
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho,
                                        name="tank")
        add_properties(tank, 'rad_s')
        tank.rad_s[:] = (0.5 * 1e-3)

        return [fluid, tank]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep())
        # integrator = EPECIntegrator(cube=RK2StepRigidBody())

        # dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 4.
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.5
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=self.ro, c0=self.co,
                        gamma=7.0),
                TaitEOS(dest='tank', sources=None, rho0=self.ro, c0=self.co,
                        gamma=7.0),
            ], real=False),
            Group(equations=[
                # ContinuityEquation(dest='fluid',
                #                    sources=['fluid', 'tank']),
                ContinuityEquation(dest='fluid',
                                   sources=['fluid', 'tank']),
                ContinuityEquation(dest='tank',
                                   sources=['tank', 'fluid']),
                MomentumEquation(dest='fluid', sources=['fluid', 'tank'],
                                 alpha=self.alpha, beta=0.0, c0=self.co,
                                 gy=-9.81),
                XSPHCorrection(dest='fluid', sources=['fluid', 'tank']),
            ]),
        ]
        return equations


if __name__ == '__main__':
    app = FluidStructureInteration()
    app.run()
    # x, y = create_fluid()
    # xb, yb = create_boundary()
    # plt.scatter(x, y)
    # plt.scatter(xb, yb)
    # xt, yt = create_boundary(1 * 1e-3)
    # xc, yc, indices = create_cube()
    # xf, yf = create_fluid_with_solid_cube()
    # plt.scatter(xt, yt)
    # plt.scatter(xc, yc)
    # plt.scatter(xf, yf)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()
