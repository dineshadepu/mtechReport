"""Water rested in a vessel

Check basic equations of SPH to throw a ball inside the vessel
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation,)
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.solver.application import Application


def create_fluid_with_solid_cube(dx=2 * 1e-3):
    x_s = 0
    x_e = 140 * 1e-3
    y_s = 0 * 1e-3
    y_e = 130 * 1e-3

    x, y = np.mgrid[x_s:x_e + 1e-9:dx, y_s:y_e + 1e-9:dx]
    x = x.ravel()
    y = y.ravel()

    indices = []
    for i in range(len(x)):
        if 60 * 1e-3 <= x[i] <= 80 * 1e-3:
            if y[i] >= 120 * 1e-3:
                indices.append(i)

    x, y = np.delete(x, indices), np.delete(y, indices)
    # print(len(x))
    return x, y


def create_fluid(dx=2 * 1e-3):
    x_s = 0
    x_e = 140 * 1e-3
    y_s = 0 * 1e-3 + dx
    y_e = 130 * 1e-3

    x, y = np.mgrid[x_s:x_e:dx, y_s:y_e + 1e-9:dx]
    x = x.ravel()
    y = y.ravel()
    # print(len(x))
    return x, y


def create_boundary(dx=2 * 1e-3):
    _dx = dx
    # make boundary particles closer
    x_s = -dx
    x_e = -dx - _dx - 1e-9
    y_s = -2 * 1e-3
    y_e = 150 * 1e-3
    x, y = np.mgrid[x_s:x_e:-_dx, y_s:y_e:_dx]
    xl = x.ravel()
    yl = y.ravel()

    x_s = 142 * 1e-3
    x_e = (142 + 2) * 1e-3
    y_s = -2 * 1e-3
    y_e = 150 * 1e-3
    x, y = np.mgrid[x_s:x_e:_dx, y_s:y_e:_dx]
    xr = x.ravel()
    yr = y.ravel()

    x_s = -dx - _dx
    x_e = (142 + 2) * 1e-3
    y_s = -dx
    y_e = -dx - _dx - 1e-9
    x, y = np.mgrid[x_s:x_e:_dx, y_s:y_e:-_dx]
    xm = x.ravel()
    ym = y.ravel()
    x, y = np.concatenate([xl, xr, xm]), np.concatenate([yl, yr, ym])
    # x, y = np.concatenate([xl, xm]), np.concatenate([yl, ym])
    # print(x)
    # print(len(x))

    return x, y


def create_cube():
    # Cube length is 20mm
    # Vessel base is 140mm
    # Cube starts at 60mm and ends at 80 mm
    # Y position of cube is 120 to 140
    x, y = np.mgrid[60 * 1e-3:80 * 1e-3 + 1e-9:1 * 1e-3, 133 * 1e-3:153 * 1e-3
                    + 1e-9:1e-3]
    x, y = x.ravel(), y.ravel()
    indices = []
    for i in range(len(x)):
        if 60 * 1e-3 < x[i] < 80 * 1e-3:
            if 133 * 1e-3 < y[i] < 153 * 1e-3:
                indices.append(0)
            else:
                indices.append(1)
        else:
            indices.append(1)

    return x, y, np.asarray(indices)


def initialize_density_fluid(x, y):
    # Taken from
    # https://www.uibk.ac.at/umwelttechnik/teaching/master/master_bergmeister.pdf
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


def initialize_mass(x, y):
    rho = initialize_density_fluid(x, y)

    m = np.zeros_like(x)
    dx = 2 * 1e-3
    m[:] = rho * dx * dx

    return m


def find_h_cube(x, y):
    pass


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class FluidStructureInteration(Application):
    def initialize(self):
        self.dx = 2 * 1e-3
        self.hdx = 1.2
        self.ro = 1000
        self.co = 2 * np.sqrt(2 * 9.81 * 130 * 1e-3)
        self.alpha = 0.1

    def create_particles(self):
        """Create the circular patch of fluid."""
        # xf, yf = create_fluid_with_solid_cube()
        xf, yf = create_fluid()
        uf = np.zeros_like(xf)
        vf = np.zeros_like(xf)
        m = initialize_mass(xf, yf)
        rho = initialize_density_fluid(xf, yf)
        h = np.ones_like(xf) * self.hdx * self.dx
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho, u=uf,
                                         v=vf, name="fluid")

        xt, yt = create_boundary(self.dx / 2.)
        ut = np.zeros_like(xt)
        vt = np.zeros_like(xt)
        m = np.ones_like(xt) * 1500 * self.dx * self.dx
        rho = np.ones_like(xt) * 1000
        h = np.ones_like(xt) * self.hdx * self.dx / 2.
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho, u=ut,
                                        v=vt, name="tank")

        return [fluid, tank]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep())

        dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
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
                ContinuityEquation(
                    dest='fluid',
                    sources=['fluid', 'tank'],),
                ContinuityEquation(
                    dest='tank',
                    sources=['fluid', 'tank'], ),
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
    # xt, yt = create_boundary(1 * 1e-3)
    # plt.scatter(xt, yt)
    # plt.scatter(x, y)
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
