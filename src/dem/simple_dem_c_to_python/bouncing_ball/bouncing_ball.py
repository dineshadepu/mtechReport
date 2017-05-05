import sys
import csv
import numpy as np
sys.path.append('../')
import pylab
import headers as headers
from grains import (prediction, interact_grains, update_acc, correction)
# from global_values import dt

# from walls import interact_walls

# Create global properties
tf = 0.00001
dt = 1e-7
maxStep = int(tf / dt)


def main():
    # Benchmark 1 data
    g1 = headers.Particle(R=0.01, x=-0.0102, vx=10, rho=2800, nu=0.2,
                          E=4.8 * 1e10)
    g2 = headers.Particle(R=0.01, x=0.0102, vx=-10, rho=2800, nu=0.2,
                          E=4.8 * 1e10)
    # g1 = headers.Particle(R=0.01, x=-0.0102, vx=10, rho=2500, nu=0.25,
    #                       E=2 * 1e10)
    # g2 = headers.Particle(R=0.01, x=0.0102, vx=-10, rho=2500, nu=0.25,
    #                       E=2 * 1e10)
    g = [g1, g2]
    time = 0
    force = []
    t = []
    for i in range(maxStep):
        # print(i)
        time += dt  # update current time step

        # predict new kinematics
        prediction(g)

        # Calculate contact forces between grains
        interact_grains(g)
        # apply_gravity(g)

        # Calculate contact forces between grains and walls
        # interact_walls(g, wallDown)
        force.append(-g[0].fx)
        t.append(time)

        # update acceleration from forces
        update_acc(g)

        # correct velocities
        correction(g)

    print("Simulation ended without errors")
    return force, t


def get_benchmark_data(filename='dataset.csv'):
    force = []
    time = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            force.append(row[1])
            time.append(row[0])

    time = np.asarray(time)
    return force, time


if __name__ == '__main__':
    force, t = main()
    t = np.asarray(t)
    t = t*1e7 - 20
    pylab.plot(t, force, label='Self')

    # Benchmark
    f_b, t_b = get_benchmark_data('dataset1.csv')
    pylab.scatter(t_b, f_b, label='benchmark')
    pylab.xlabel(r'Time ()$\alpha > \beta$')
    pylab.ylabel("Normal contact force (N)")
    pylab.xlim(0, 40)
    pylab.legend(loc='upper left')
    pylab.show()
