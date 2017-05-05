import sys
sys.path.append('../')
import numpy as np
import headers
import pylab
import numpy as np
from grains import (prediction, update_acc, correction)
from walls import interact_walls

# Create global properties
tf = 0.0013
dt = 1e-6
maxStep = int(tf / dt)


def main():
    # benchmark 2 data
    g1 = headers.Particle(R=0.1, y=0.100002, vy=-0.2, rho=1800, nu=0.35, E=4 *
                          1e10)
    wallDown = headers.Wall(pos=0.0, E=4 * 1e10, nu=0.35)
    # g1 = headers.Particle(R=0.1, y=0.1001, vy=-0.2, rho=2699, nu=0.3,
    #                       E=7 * 1e10)
    # wallDown = headers.Wall(pos=0.0, E=7 * 1e10, nu=0.3)
    g = [g1]
    time = 0
    force = []
    t = []
    for i in range(maxStep):
        # print(i)
        time += dt  # update current time step

        # predict new kinematics
        prediction(g)

        # Calculate contact forces between grains and walls
        interact_walls(g, wallDown)
        force.append(g[0].fy)
        t.append(time)

        # update acceleration from forces
        update_acc(g)

        # correct velocities
        correction(g)

    print("Simulation ended without errors")
    return np.asarray(force), np.asarray(t)


if __name__ == '__main__':
    force, t = main()
    pylab.plot(t * 1e6, force)
    pylab.show()
