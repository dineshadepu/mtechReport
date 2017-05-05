import sys
import numpy as np
sys.path.append('../')
import headers
import pylab
import numpy as np
from grains import (prediction, update_acc, correction, interact_grains)
from walls import interact_walls

# Create global properties
tf = 0.0013
dt = 1e-6
maxStep = int(tf / dt)


def main():
    # benchmark 2 data
    # al oxide
    g1 = headers.Particle(R=0.0025, y=0.003, vy=-3.9, rho=4000, nu=0.23,
                          E=3.8 * 1e11, e=0.25)
    g2 = headers.Particle(R=0.0025, y=-0.003, vy=3.9, rho=4000, nu=0.23,
                          E=3.8 * 1e11, e=0.25, fixed=True)
    # wallDown = headers.Wall(pos=0.0, E=3.8 * 1e11, nu=0.23)
    # g1 = headers.Particle(R=0.1, y=0.100001, vy=-0.2, rho=2699, nu=0.3,
    #                       E=7 * 1e10)
    # wallDown = headers.Wall(pos=0.0, E=7 * 1e10, nu=0.3)
    # g = [g1]
    g = [g1, g2]
    time = 0

    # starting velocity of g[0]
    g_0_vy_start = g[0].vy
    for i in range(maxStep):
        # print(i)
        time += dt  # update current time step

        # predict new kinematics
        prediction(g)

        # Calculate contact forces between grains
        interact_grains(g)

        # Calculate contact forces between grains and walls
        # interact_walls(g, wallDown)

        # update acceleration from forces
        update_acc(g)

        # correct velocities
        correction(g)

    print(g_0_vy_start / -g[0].vy)
    print("Simulation ended without errors")


if __name__ == '__main__':
    main()
