"""Water rested in a vessel, and a cube falls into it

Check basic equations of SPH to throw a ball inside the vessel
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_wcsph, get_particle_array_rigid_body
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation,)
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision,
                                  SolidFluidForce,
                                  NumberDensity, RigidBodyMoments,
                                  RigidBodyMotion, RK2StepRigidBody,
                                  PressureRigidBody)


def create_boundary():
    dx = 2

    # bottom particles in tank
    xb = np.arange(-2*dx, 140+2*dx, dx)
    yb = np.arange(-2*dx, 0, dx)
    xb, yb = np.meshgrid(xb, yb)
    xb = xb.ravel()
    yb = yb.ravel()

    xl = np.arange(-2*dx, 0, dx)
    yl = np.arange(0, 150, dx)
    xl, yl = np.meshgrid(xl, yl)
    xl = xl.ravel()
    yl = yl.ravel()

    xr = np.arange(140, 140+2*dx, dx)
    yr = np.arange(0, 150, dx)
    xr, yr = np.meshgrid(xr, yr)
    xr = xr.ravel()
    yr = yr.ravel()

    x = np.concatenate([xl, xb, xr])
    y = np.concatenate([yl, yb, yr])

    return x*1e-3, y*1e-3


def create_fluid():
    dx = 2
    # xf = np.arange(dx/2., 140-dx/2., dx)
    # yf = np.arange(dx/2., 130+dx/2., dx)
    xf = np.arange(0, 140, dx)
    yf = np.arange(0, 130, dx)
    xf, yf = np.meshgrid(xf, yf)
    xf = xf.ravel()
    yf = yf.ravel()

    return xf*1e-3, yf*1e-3


def create_cube(dx=1):
    x = np.arange(60, 80, dx)
    y = np.arange(131, 151, dx)
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()

    return x*1e-3, y*1e-3


def get_density(y):
    c_0 = 2 * np.sqrt(2 * 9.81 * 130 * 1e-3)
    rho_0 = 1000
    height_water_clmn = 130 * 1e-3
    gamma = 7.
    _tmp = gamma / (rho_0 * c_0**2)

    rho = np.zeros_like(y)
    for i in range(len(rho)):
        p_i = rho_0 * 9.81 * (height_water_clmn - y[i])
        rho[i] = rho_0 * (1 + p_i * _tmp)**(1. / gamma)
    return rho


def geometry():
    # please run this function to know how
    # geometry looks like
    x_tank, y_tank = create_boundary()
    x_fluid, y_fluid = create_fluid()
    plt.scatter(x_fluid, y_fluid)
    plt.scatter(x_tank, y_tank)
    plt.axes().set_aspect('equal', 'datalim')
    print("done")
    plt.show()


class FluidStructureInteration(Application):
    def initialize(self):
        self.dx = 2 * 1e-3
        self.hdx = 1.2
        self.ro = 1000
        self.m = 1000 * self.dx * self.dx
        self.co = 2 * np.sqrt(2 * 9.81 * 130 * 1e-3)
        self.alpha = 0.1

    def create_particles(self):
        """Create the circular patch of fluid."""
        # xf, yf = create_fluid_with_solid_cube()
        xf, yf = create_fluid()
        rho = get_density(yf)
        m = rho[:] * self.dx * self.dx
        rho = np.ones_like(xf) * self.ro
        h = np.ones_like(xf) * self.hdx * self.dx
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho,
                                         name="fluid")

        xt, yt = create_boundary()
        m = np.ones_like(xt) * 1000 * self.dx * self.dx
        rho = np.ones_like(xt) * 1000
        h = np.ones_like(xt) * self.hdx * self.dx
        rad_s = np.ones_like(xt) * 2 / 2. * 1e-3
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho,
                                        name="tank", rad_s=rad_s)

        dx = 1
        xc, yc = create_cube(1)
        m = np.ones_like(xc) * 2120 * dx*1e-3 * dx*1e-3
        rho = np.ones_like(xc) * 2120
        h = np.ones_like(xc) * self.hdx * self.dx
        rad_s = np.ones_like(xc) * dx / 2. * 1e-3
        cube = get_particle_array_rigid_body(x=xc, y=yc, h=h, m=m, rho=rho,
                                             rad_s=rad_s,
                                             name="cube")
        return [fluid, tank, cube]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep(), tank=WCSPHStep(),
                                    cube=RK2StepRigidBody())

        dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
        print("DT: %s" % dt)
        tf = 0.5
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                    BodyForce(dest='cube', sources=None, gy=-9.81),
                    # NumberDensity(dest='cube', sources=['cube']),
                    ], real=False),

            Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=self.ro, c0=self.co,
                        gamma=7.0),
                TaitEOS(dest='tank', sources=None, rho0=self.ro, c0=self.co,
                        gamma=7.0),
            ], real=False),

            Group(equations=[
                ContinuityEquation(
                    dest='fluid',
                    sources=['fluid', 'tank', 'cube'],),
                ContinuityEquation(
                    dest='tank',
                    sources=['fluid', 'tank', 'cube'], ),
                MomentumEquation(dest='fluid', sources=['fluid', 'tank'],
                                 alpha=self.alpha, beta=0.0, c0=self.co,
                                 gy=-9.81),
                SolidFluidForce(dest='fluid', sources=['cube'],),
                # PressureRigidBody(dest='fluid', sources=['cube'],
                #                   rho0=1500),

                XSPHCorrection(dest='fluid', sources=['fluid', 'tank']),
            ]),
            Group(equations=[RigidBodyCollision(dest='cube', sources=['tank'],
                                                kn=1e5)]),
            Group(equations=[RigidBodyMoments(dest='cube', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='cube', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = FluidStructureInteration()
    app.run()
    # geometry()
