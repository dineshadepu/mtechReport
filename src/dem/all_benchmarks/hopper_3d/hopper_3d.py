"""100 spheres falling inside hopper

Check the complete molecular dynamics code
"""
from __future__ import print_function
import numpy as np
import math
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

# import matplotlib.pyplot as plt

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group
from pysph.sph.molecular_dynamics import (
    DEMStep,
    get_particle_array_dem,
    LinearSpringForceParticleParticle,
    # MakeForcesZero,
    BodyForce, )
from pysph.solver.application import Application


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class FluidStructureInteration(Application):
    def initialize(self):
        self.dx = 1
        radius_of_par = 0.5
        _m = 4. / 3. * np.pi * radius_of_par**3 * 2800
        self.m_eff = _m / 2.

    def sand_geometry(self):
        radius_of_par = 0.5
        x = np.arange(-4, 4, 2*radius_of_par)
        y = np.arange(-4, 4, 2*radius_of_par)
        z = np.arange(6, 30, 2*radius_of_par)
        x, y, z = np.meshgrid(x, y, z)
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        print(z)
        # fig = pylab.figure()
        # ax = Axes3D(fig)
        # ax.scatter(x, y, z, s=1000)
        return x, y, z

    def hopper_geometry(self, hopper_small_par_radius=0.5):
        # small radius cone
        r_s = 3.
        # big radius cone
        r_b = 8

        height = 10.
        tan_theta = height / (r_b - r_s)

        x_total = np.asarray([])
        y_total = np.asarray([])
        z_total = np.asarray([])

        z = 0
        increasing_rad = r_s
        while increasing_rad <= r_b:
            d_theta = 2*hopper_small_par_radius / increasing_rad
            theta = np.arange(0, 2*np.pi, d_theta)

            x_total = np.concatenate([x_total, increasing_rad *
                                      np.cos(theta)])
            y_total = np.concatenate([y_total, increasing_rad *
                                      np.sin(theta)])
            z_total = np.concatenate([z_total, np.ones_like(theta)*z])

            z = z + 2 * hopper_small_par_radius
            increasing_rad = r_b - (height - z) / tan_theta

        # fig = pylab.figure()
        # ax = Axes3D(fig)
        # ax.scatter(x_total, y_total, z_total, s=1000)
        # pyplot.show()
        return x_total, y_total, z_total

    def plane_wall(self):
        x = np.arange(-50, 50, 1)
        y = np.arange(-50, 50, 1)
        x, y = np.meshgrid(x, y)
        x, y = x.ravel(), y.ravel()
        z = np.ones_like(x) * -10
        return x, y, z

    def create_particles(self):
        radius_of_par = 0.5
        x, y, z = self.sand_geometry()
        R = np.ones_like(x) * radius_of_par
        _m = 4. / 3. * np.pi * radius_of_par**3 * 2800
        m = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2. / 5. * _m * radius_of_par**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * radius_of_par
        sand = get_particle_array_dem(x=x, y=y, z=z, m=m,
                                      m_inverse=m_inverse, R=R, h=h,
                                      I_inverse=I_inverse,
                                      name="sand")

        x, y, z = self.hopper_geometry()
        m = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        R = np.ones_like(x) * radius_of_par
        h = np.ones_like(x) * radius_of_par
        _I = 2. / 5. * _m * radius_of_par**2
        I_inverse = np.ones_like(x) * 1. / _I
        wall = get_particle_array_dem(x=x, y=y, z=z, m=m,
                                      m_inverse=m_inverse, R=R, h=h,
                                      I_inverse=I_inverse,
                                      name="wall")

        x, y, z = self.plane_wall()
        m = np.ones_like(x) * _m
        m_inverse = np.ones_like(x) * 1. / _m
        R = np.ones_like(x) * radius_of_par
        h = np.ones_like(x) * radius_of_par
        _I = 2. / 5. * _m * radius_of_par**2
        I_inverse = np.ones_like(x) * 1. / _I
        plane = get_particle_array_dem(x=x, y=y, z=z, m=m,
                                       m_inverse=m_inverse, R=R, h=h,
                                       I_inverse=I_inverse,
                                       name="plane")
        return [sand, wall, plane]

    def create_solver(self):
        kernel = CubicSpline(dim=3)

        integrator = EPECIntegrator(sand=DEMStep())

        dt = 1e-4
        print("DT: %s" % dt)
        tf = 30
        solver = Solver(kernel=kernel, dim=3, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='sand', sources=None, gz=-9.81),
                LinearSpringForceParticleParticle(
                    dest='sand', sources=['sand', 'wall', 'plane'], k=1e8,
                    ln_e=abs(np.log(0.2)), m_eff=self.m_eff, mu=.5),
                # MakeForcesZero(dest='sand', sources=None)
            ]),
        ]
        return equations

    def plot_geometry(self):
        xh, yh, zh = self.hopper_geometry()
        xs, ys, zs = self.sand_geometry()

        fig = pylab.figure()
        ax = Axes3D(fig)
        ax.scatter(xh, yh, zh, s=1000)
        ax.scatter(xs, ys, zs, s=1000)
        pyplot.show()

    # def pre_step(self, solver):
    #     solver.dump_output()


if __name__ == '__main__':
    app = FluidStructureInteration()
    # app.create_particles()
    app.run()
    # app.plot_geometry()

#  LocalWords:  SummationDensityShepardFilter
