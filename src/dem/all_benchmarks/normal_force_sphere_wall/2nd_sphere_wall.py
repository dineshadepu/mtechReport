import numpy as np
import matplotlib.pyplot as plt


# al alloy properties
r = 0.10
rho = 2699
m = np.pi * 4. / 3. * r**3 * rho
E = 7 * 1e10
nu = 0.3
k = 4. / 3. * E * np.sqrt(r) / (1 - nu**2)

# mg alloy properties
# r = 0.10
# rho = 1800
# m = np.pi * 4. / 3. * r**3 * rho
# E = 4 * 1e10
# nu = 0.35
# k = 4. / 3. * E * np.sqrt(r) / (1 - nu**2)


def create_particles():
    x = np.array([-0.1001, 0.])
    y = np.array([0., 0.])
    u = np.array([0.2, 0])
    v = np.array([0., 0.])
    return x, y, u, v


def find_forces(x, y, u, v):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dist = np.sqrt((dx)**2 + (dy)**2)
    overlap = r - dist
    fx = np.asarray([0., 0.])
    fy = np.asarray([0., 0.])
    if overlap > 0:
        fn = k * overlap**(3./2.)
        nx = dx / dist
        ny = dy / dist
        fx[0] = -fn*nx
        fx[1] = 0
        fy[0] = -fn*ny
        fy[1] = 0

    return fx, fy, overlap


def integrate(x, y, u, v, fx, fy, dt):
    # u = u + fx * dt
    # v = v + fy * dt
    # x = x + u * dt
    # y = y + v * dt
    u += fx * dt / m
    v += fy * dt / m
    x += u * dt
    y += v * dt


def simulation(tf=0.002, dt=1e-6):
    x, y, u, v = create_particles()

    # Integrate the system
    t = np.arange(0, tf, dt)
    req = []
    t_p = []
    for i in range(len(t)):
        fx, fy, overlap = find_forces(x, y, u, v)
        if abs(fx[0]) > 0:
            t_p.append(t[i])
            # t_p.append(overlap)
            req.append(-fx[0])
        integrate(x, y, u, v, fx, fy, dt)
    print(max(req))
    # substact the minimum value from t_p to get good graph as in benchmark
    t_p = np.asarray(t_p)
    t_p = (t_p - min(t_p)) * 1e6
    plt.plot(t_p, req)
    plt.show()


simulation()
