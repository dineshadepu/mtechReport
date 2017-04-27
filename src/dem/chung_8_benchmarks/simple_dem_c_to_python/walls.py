from global_values import nu
import numpy as np


def compute_force_lower_wall(i, g, wall):
    dn = wall.pos - (g[i].y - g[i].R)
    tmp1 = 4. * np.sqrt(g[i].R) / 3.
    tmp_1 = (1. - g[i].nu * g[i].nu) / g[i].E
    tmp_2 = (1. - wall.nu * wall.nu) / wall.E
    E_eff = 1. / (tmp_1 + tmp_2)
    kn = E_eff * tmp1
    kt = 2. / 7. * kn

    if dn > 0:
        vn = g[i].vy
        # vt = g[i].vt

        fn = kn * dn**(3./2.) - nu * vn

        g[i].fy += fn


def interact_walls(grains, wallDown):
    for i in range(len(grains)):
        compute_force_lower_wall(i, grains, wallDown)
