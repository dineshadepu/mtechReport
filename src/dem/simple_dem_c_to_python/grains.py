from global_values import dt, grav, nu, mu
import numpy as np


def prediction(grains):
    for g in grains:
        if g.fixed is False:
            g.x += dt * g.vx + 0.5 * dt * dt * g.ax
            g.y += dt * g.vy + 0.5 * dt * dt * g.ay
            g.vx += 0.5 * dt * g.ax
            g.vy += 0.5 * dt * g.ay

            g.ang += dt * g.angv + 0.5 * dt * dt * g.anga
            g.angv += 0.5 * dt * g.anga

        # Zero forces
        g.fx = 0
        g.fy = 0.0
        g.t = 0.0
        g.p = 0.0


def interparticle_force(g, a, b):
    if a > b:
        # particle centre coordinate component differences
        x_ab = g[b].x - g[a].x
        y_ab = g[b].y - g[a].y

        # Particle centre distance
        dist = np.sqrt(x_ab * x_ab + y_ab * y_ab)

        # size of overlap
        dn = (g[a].R + g[b].R) - dist
        if dn > 0:
            xn = x_ab / dist
            yn = y_ab / dist
            xt = -yn
            yt = xn

            # compute the velocity of the contact
            vx_ab = g[a].vx - g[b].vx
            vy_ab = g[a].vy - g[b].vy
            vn = vx_ab * xn + vy_ab * yn
            vt = vx_ab * xt + vy_ab * yt - (g[a].R * g[a].angv + g[b].R *
                                            g[b].angv)

            # find contact parameters
            r_eff = (g[a].R * g[b].R) / (g[a].R + g[b].R)
            m_eff = (g[a].m * g[b].m) / (g[a].m + g[b].m)
            tmp_1 = (1. - g[a].nu * g[a].nu) / g[a].E
            tmp_2 = (1. - g[b].nu * g[b].nu) / g[b].E
            E_eff = 1. / (tmp_1 + tmp_2)

            kn = 4./3. * np.sqrt(r_eff) * E_eff
            kt = 2./7. * kn

            # Compute force in local axes
            fn = -kn * dn**(3./2.) - nu * vn

            # rotation
            ft = abs(kt * vt)
            if ft > mu * fn:
                ft = mu * fn
            if vt > 0:
                ft = -ft

            # calculate sum of forces on a and b in global coordinates
            g[a].fx += fn * xn
            g[a].fy += fn * yn
            g[a].t += -ft * g[a].R
            g[a].p += fn
            g[b].fx -= fn * xn
            g[b].fy -= fn * yn
            g[b].p += fn
            g[b].t += -ft * g[b].R


def interact_grains(grains):
    # loop through particle a
    for a in range(len(grains)):
        # Loop through particle b
        for b in range(len(grains)):
            interparticle_force(grains, a, b)


def update_acc(grains):
    for g in grains:
        g.ax = g.fx / g.m
        g.ay = g.fy / g.m
        g.anga = g.t / g.I


def apply_gravity(grains):
    for g in grains:
        g.ay += - grav


def correction(grains):
    for g in grains:
        if g.fixed is False:
            g.vx += 0.5 * dt * g.ax
            g.vy += 0.5 * dt * g.ay
            g.angv += 0.5 * dt * g.anga
