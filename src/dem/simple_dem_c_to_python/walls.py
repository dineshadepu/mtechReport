import numpy as np


def compute_force_lower_wall(i, g, wall):
    dn = wall.pos - (g[i].y - g[i].R)

    if dn > 0:
        vn = g[i].vy
        # vt = g[i].vt

        tmp1 = 4. * np.sqrt(g[i].R) / 3.
        tmp_1 = (1. - g[i].nu * g[i].nu) / g[i].E
        tmp_2 = (1. - wall.nu * wall.nu) / wall.E
        E_eff = 1. / (tmp_1 + tmp_2)
        kn = E_eff * tmp1
        kt = 2. / 7. * kn

        alpha_1 = -np.log(g[i].e) * np.sqrt(5. / (
            np.log(g[i].e)**2 + np.pi**2))
        eta_tmp = alpha_1 * np.sqrt(g[i].m/2 * kn)
        eta_n = eta_tmp * dn**(1./4.)

        fn = kn * dn**(3./2.) - eta_n * vn

        g[i].fy += fn


def interact_walls(grains, wallDown):
    for i in range(len(grains)):
        compute_force_lower_wall(i, grains, wallDown)
