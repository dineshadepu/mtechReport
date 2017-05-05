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
    # print(len(x))
    return x, y


def create_boundary(dx=2 * 1e-3):
    # Bottom of the tank
    x = np.arange(0, 150 * 1e-3 + 1e-9, dx)
    y = np.arange(-dx, -2 * dx - 1e-9, -dx)

    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()

    # Left particles of the tank
    xl = np.arange(-dx, -2 * dx - 1e-9, -dx)
    yl = np.arange(-2 * dx, 140 * 1e-3 + 1e-9, dx)

    xl, yl = np.meshgrid(xl, yl)
    xl, yl = xl.ravel(), yl.ravel()

    # Right particle of the tank
    xr = np.arange(150 * 1e-3 + dx, 150 * 1e-3 + 2 * dx + 1e-9, dx)
    yr = np.arange(-2 * dx, 140 * 1e-3 + 1e-9, dx)

    xr, yr = np.meshgrid(xr, yr)
    xr, yr = xr.ravel(), yr.ravel()

    # concatenate all parts of tank
    x = np.concatenate([x, xl, xr])
    y = np.concatenate([y, yl, yr])
    return x, y


def create_cube():
    # Cube length is 20mm
    # Vessel base is 140mm
    # Cube starts at 60mm and ends at 80 mm
    # Y position of cube is 120 to 140
    x = np.arange(-10*1e-3, 10*1e-3+1e-9, 1e-3)
    y = np.arange(-10*1e-3, 10*1e-3+1e-9, 1e-3)
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()
    x = x + 75 * 1e-3
    y = y + 130 * 1e-3
    indices = []
    for i in range(len(x)):
        if 65 * 1e-3 < x[i] < 84 * 1e-3:
            if 120.5 * 1e-3 < y[i] < 139 * 1e-3:
                indices.append(0)
            else:
                indices.append(1)
        else:
            indices.append(1)

    # Check if indices are correct
    # print(len(x))
    # print(len(indices))
    # x_new = []
    # y_new = []

    # for i in range(len(x)):
    #     if indices[i] == 1:
    #         x_new.append(x[i])
    #         y_new.append(y[i])

    # plt.scatter(x_new, y_new)
    # plt.show()

    return x, y, np.asarray(indices)
    # return x, y, []


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
        xf, yf = create_fluid_with_solid_cube()
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
        m = np.ones_like(xt) * 2120 * self.dx * self.dx
        rho = np.ones_like(xt) * 1000
        h = np.ones_like(xt) * self.hdx * self.dx / 2.
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho, u=ut,
                                        v=vt, name="tank")
        add_properties(tank, 'rad_s')
        tank.rad_s[:] = (0.5 * 1e-3)

        xc, yc, indices = create_cube()
        _m = 2120 * self.dx * self.dx / 2.
        m = np.ones_like(xc) * _m
        h = np.ones_like(xc) * self.hdx * self.dx / 2.
        rho = np.ones_like(xc) * 2120
        cube = get_particle_array_rigid_body(name="cube", x=xc, y=yc, m=m, h=h,
                                             rho=rho)
        add_properties(cube, 'indices')
        cube.indices[:] = indices[:]
        add_properties(cube, 'rad_s')
        cube.rad_s[:] = 0.5 * 1e-3
        add_properties(
            cube,
            'tang_disp_x',
            'tang_disp_y',
            'tang_disp_z',
            'tang_disp_x0',
            'tang_disp_y0',
            'tang_disp_z0',
            'tang_velocity_x',
            'tang_velocity_y',
            'tang_velocity_z', )

        return [fluid, tank, cube]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep(), cube=RK2StepRigidBody())
        # integrator = EPECIntegrator(cube=RK2StepRigidBody())

        dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 4.
        print("DT: %s" % dt)
        tf = 0.5
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            # Group(equations=[
            #     BoundaryParticleNumberDenstiy(dest='cube', sources=['cube'])
            # ]),
            # Group(equations=[
            #     SummationDensityRigidBody(dest='fluid', sources=['cube'],
            #                               rho0=1000)
            # ]),
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
                                   sources=['fluid', 'tank', 'cube']),
                ContinuityEquation(dest='tank',
                                   sources=['tank', 'fluid']),
                MomentumEquation(dest='fluid', sources=['fluid', 'tank'],
                                 alpha=self.alpha, beta=0.0, c0=self.co,
                                 gy=-9.81),
                SolidForceOnFluid(dest='fluid', sources=['cube']),
                XSPHCorrection(dest='fluid', sources=['fluid', 'tank']),
            ]),
            Group(equations=[
                BodyForce(dest='cube', sources=None, gy=-9.81),
                FluidForceOnSolid(dest='cube', sources=['fluid', 'tank']),
                RigidBodyCollision(
                    dest='cube',
                    sources=['tank'],
                    kn=1e5,
                    en=0.5, )
            ]),
            Group(equations=[RigidBodyMoments(dest='cube', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='cube', sources=None)]),
        ]
        return equations


if __name__ == '__main__':
    app = FluidStructureInteration()
    app.run()
    # xt, yt = create_boundary(1 * 1e-3)
    # xc, yc, indices = create_cube()
    # xf, yf = create_fluid_with_solid_cube()
    # plt.scatter(xt, yt)
    # plt.scatter(xc, yc)
    # plt.scatter(xf, yf)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()
