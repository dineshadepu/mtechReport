import numpy as np
import pylab
import csv

rad = 0.010
density = 2800.
m = 4. / 3. * np.pi * rad**3 * density
m_eff = m / 2.
E = 4.8 * 1e10
sigma = 0.20
coeff_of_rest = 1
k_n = (np.sqrt(2 * rad) * E) / (3 * (1 - sigma**2))
alpha_1 = -np.log(coeff_of_rest) * np.sqrt(5 / (
    np.log(coeff_of_rest)**2 + np.pi**2))
eta_tmp = alpha_1 * np.sqrt(m_eff * k_n)


class Particle(object):
    """A DEM particle
    """

    def __init__(self, rad=0, x=0, y=0, vx=0, vy=0, omega=0, fx=0, fy=0):
        super(Particle, self).__init__()
        self.rad = rad
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.fx = fx
        self.fy = fy


def create_particles():
    par1 = Particle(0.01, -0.015, 0, 10, 0, 0, 0)
    par2 = Particle(0.01, 0.015, 0, -10, 0, 0, 0)
    return par1, par2


def find_forces(par1, par2):
    # x[0] is left particle coming with positive velocity
    # x[1] is right particle coming with negative velocity
    dx = par2.x - par1.x
    dy = par2.y - par1.y

    # find distance between particles
    dist = np.sqrt(dx**2 + dy**2)

    # Find overlap amount
    overlap = par1.rad + par2.rad - dist

    # check if particles are in contact
    if overlap > 0:
        # print("inside")
        # unit vectors
        n_x = dx / dist
        n_y = dy / dist

        # Linear velocity difference
        dv_x = par1.vx - par2.vx
        dv_y = par1.vy - par2.vy

        # Find distance from centre to contact point
        L_1 = (dist**2 + par1.rad**2 - par2.rad**2) / (2 * dist)
        L_2 = L_1 - dist

        # Angular velocity at contact difference
        omega_dv = np.cross([0, 0, L_1 * par1.omega + L_2 * par2.omega],
                            [n_x, n_y])

        # Total relative velocity including angular component
        dv_x = dv_x + omega_dv[0]
        dv_y = dv_y + omega_dv[1]

        v_n_mag = np.dot([dv_x, dv_y], [n_x, n_y])

        v_n_x = v_n_mag * n_x
        v_n_y = v_n_mag * n_y

        v_t_x = dv_x - v_n_x
        v_t_y = dv_y - v_n_y

        v_t_mag = np.sqrt(v_t_x**2 + v_t_y**2)

        # tangential unit vector
        t_x = 0
        t_y = 0
        if v_t_mag != 0:
            t_x = v_t_x / v_t_mag
            t_y = v_t_y / v_t_mag

        eta_n = eta_tmp * overlap**(1. / 4.)
        tmp1 = -k_n * overlap**(3. / 2.)
        tmp2 = -eta_n * v_n_mag

        par1.fx = (tmp1 + tmp2) * n_x
        par1.fy = (tmp1 + tmp2) * n_y

        par2.fx = -(tmp1 + tmp2) * n_x
        par2.fy = -(tmp1 + tmp2) * n_y
        return True
    else:
        return False


def integrate(par1, par2):
    par1.vx = par1.vx + par1.fx / m * dt
    par1.vy = par1.vy + par1.fy / m * dt

    par1.x = par1.x + par1.vx * dt
    par1.y = par1.y + par1.vy * dt

    par2.vx = par2.vx + par2.fx / m * dt
    par2.vy = par2.vy + par2.fy / m * dt

    par2.x = par2.x + par2.vx * dt
    par2.y = par2.y + par2.vy * dt


def simulate(tf, dt):
    par1, par2 = create_particles()
    force = []
    t = []
    n = int(tf / dt)
    tf = 0
    for i in range(0, n):
        p = find_forces(par1, par2)
        integrate(par1, par2)
        if p:
            force.append(-par1.fx)
            t.append(tf)
        tf = tf + dt
    return force, t


def get_benchmark_data(filename='dataset.csv'):
    force = []
    time = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            force.append(row[1])
            time.append(row[0])

    return force, time


dt = 1e-7
if __name__ == '__main__':
    f_b, time = get_benchmark_data('dataset1.csv')
    print("done")
    force, t = simulate(0.005, dt=dt)
    t = np.asarray(t)
    t = t - t[0]
    pylab.plot(t * 1e6, force, label='Self')
    pylab.scatter(time, f_b, label='benchmark')
    pylab.xlabel(r'Time ()$\alpha > \beta$')
    pylab.ylabel("Normal contact force (N)")
    pylab.legend(loc='upper left')
    pylab.show()
    pylab.savefig("Normal_contact_force_vs_time_two_spheres_colliding.png")
