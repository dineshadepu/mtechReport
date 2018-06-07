"""DCDEM benchmark using single sphere approach
remember for preq=5 or 1

Check the complete molecular dynamics code
"""
from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group
from pysph.sph.molecular_dynamics import (
    # MakeForcesZero,
    DEMStep,
    get_particle_array_dem,
    BodyForce,
    HertzSpringForceParticleParticle,
    HertzSpringForceParticleWall,)
from pysph.solver.application import Application
from centre_of_mass_post_process_dcdem import post_processing as pp

wall_time = 0.05


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


def create_wall():
    x = np.array([0, 13 * 1e-2, 26 * 1e-2])
    y = np.array([13 * 1e-2, 0, 13 * 1e-2])
    return x, y


def geometry():
    x = np.arange(0.5, 6, 1)
    y = np.ones_like(x)
    x_6 = np.concatenate([x, x, x])
    y_6 = np.concatenate([y * 0.5, y * 2.5, y * 4.5])

    x = np.arange(1, 6, 1)
    y = np.ones_like(x)
    x_5 = np.concatenate([x, x, x])
    y_5 = np.concatenate([y * 1.5, y * 3.5, y * 5.5])

    x = np.concatenate([x_6, x_5]) * 1e-2
    y = np.concatenate([y_6, y_5]) * 1e-2
    return x, y


class FluidStructureInteration(Application):
    def initialize(self):
        r = 0.5 * 1e-2
        # self._m = np.pi * r * r * 2.7 * 1e3
        self._m = 4. / 3. * np.pi * r * r * r * 2.7 * 1e3
        self.m_eff = self._m / 2

    def create_particles(self):
        x, y = geometry()
        r = 0.5 * 1e-2
        R = np.ones_like(x) * r
        _m = 4. / 3. * np.pi * r * r * r * 2.7 * 1e3
        # _m = np.pi * r * r * 2.7 * 1e3
        m = np.ones_like(x) * _m
        E = np.ones_like(x) * 69 * 1e9
        nu = np.ones_like(x) * 0.3
        m_inverse = np.ones_like(x) * 1. / _m
        _I = 2. / 5. * _m * r**2
        # _I = _m * r**2
        I_inverse = np.ones_like(x) * 1. / _I
        h = np.ones_like(x) * r

        cylinders = get_particle_array_dem(
            x=x, y=y, m=m, E=E, nu=nu, m_inverse=m_inverse, R=R, h=h,
            I_inverse=I_inverse, name="cylinders")

        # tank
        x = np.array([0, 13 * 1e-2, 26 * 1e-2])
        y = np.array([13 * 1e-2, 0, 13 * 1e-2])
        nx = np.array([1, 0, -1])
        ny = np.array([0, 1, 0])
        E = np.ones_like(x) * 30 * 1e8
        nu = np.ones_like(x) * 0.3
        h = np.ones_like(x) * 13 * 1e-2

        tank = get_particle_array_dem(x=x, y=y, E=E, nu=nu, h=h, nx=nx, ny=ny,
                                      name="tank")

        x = np.array([6 * 1e-2])
        y = np.array([3 * 1e-2])
        nx = np.array([-1])
        ny = np.array([0])
        E = np.ones_like(x) * 30 * 1e8
        nu = np.ones_like(x) * 0.3
        h = np.ones_like(x) * 1.5 * 1e-2

        t_wall = get_particle_array_dem(x=x, y=y, E=E, nu=nu, h=h, nx=nx,
                                        ny=ny, name="t_wall")

        # additional properties for equations
        # self.m_eff = (_m + mw[0]) / (_m * mw[0])
        return [cylinders, tank, t_wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(cylinders=DEMStep(), t_wall=DEMStep())

        dt = 1e-5
        # dt = 1e-6
        print("DT: %s" % dt)
        tf = 0.5 + wall_time
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, output_at_times=[wall_time+0.1,
                                                wall_time+0.3, wall_time+0.5],
                        pfreq=100000)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='cylinders', sources=None, gy=-9.81),
                HertzSpringForceParticleParticle(
                    dest='cylinders', sources=['cylinders'],
                    ln_e=abs(np.log(0.1)), mu=.250),
                HertzSpringForceParticleWall(dest='cylinders',
                                             sources=['tank'],
                                             ln_e=abs(np.log(0.1)), mu=.450),

                # temporary wall ignore this
                HertzSpringForceParticleWall(dest='cylinders',
                                             sources=['t_wall'],
                                             ln_e=abs(np.log(0.2)), mu=.450),
            ]),
        ]
        return equations

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        T = wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'temp_w':
                    break
            pa.y += 12 * 1e-2

    def post_processing(self):
        pp("dcdem_hertz_single_sphere_output/")
    # def pre_step(self, solver):
    #     solver.dump_output()


if __name__ == '__main__':
    app = FluidStructureInteration()
    # app.run()
    app.post_processing()
    # x, y = geometry()
    # xw, yw = create_wall()
    # plt.scatter(x, y)
    # plt.scatter(xw, yw)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()

#  LocalWords:  LinearSpringForceParticleWall
