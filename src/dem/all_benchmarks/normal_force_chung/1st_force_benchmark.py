import numpy as np
import matplotlib.pyplot as plt


# glass properties
# r = 0.010
# rho = 2800
# m = np.pi * 4. / 3. * r**3 * rho
# E = 4.8 * 1e10
# nu = 0.2
# k = 2. / 3. * E * np.sqrt(r / 2.) / (1 - nu**2)

# Limestone props
r = 0.010
rho = 2500
m = np.pi * 4. / 3. * r**3 * rho
E = 2 * 1e10
nu = 0.25
k = 2. / 3. * E * np.sqrt(r / 2.) / (1 - nu**2)


def create_particles():
    x = np.array([-0.010, 0.011])
    y = np.array([0., 0.])
    u = np.array([10., -10.])
    v = np.array([0., 0.])
    return x, y, u, v


def find_forces(x, y, u, v):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dist = np.sqrt((dx)**2 + (dy)**2)
    overlap = 2 * r - dist
    fx = np.asarray([0., 0.])
    fy = np.asarray([0., 0.])
    if overlap > 0:
        fn = k * overlap**(3./2.)
        nx = dx / dist
        ny = dy / dist
        fx[0] = -fn*nx
        fx[1] = fn*nx
        fy[0] = -fn*ny
        fy[1] = fn*ny

    return fx, fy


def integrate(x, y, u, v, fx, fy, dt):
    # u = u + fx * dt
    # v = v + fy * dt
    # x = x + u * dt
    # y = y + v * dt
    u += fx * dt / m
    v += fy * dt / m
    x += u * dt
    y += v * dt


def simulation(tf=0.0002, dt=1e-7):
    x, y, u, v = create_particles()

    # Integrate the system
    t = np.arange(0, tf, dt)
    req = []
    for i in range(len(t)):
        fx, fy = find_forces(x, y, u, v)
        req.append(fx[0])
        integrate(x, y, u, v, fx, fy, dt)
    plt.plot(t, req)
    plt.show()


simulation()
