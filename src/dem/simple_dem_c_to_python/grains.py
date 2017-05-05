from global_values import dt, grav
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


def interparticle_force_hamid(g, a, b):
    if a > b:
        x_ab = g[b].x - g[a].x
        y_ab = g[b].y - g[a].y
        dist = np.sqrt(x_ab * x_ab + y_ab * y_ab)

        # size of overlap
        dn = (g[a].R + g[b].R) - dist

        if dn > 0:
            # normal vector
            nx = x_ab / dist
            ny = y_ab / dist

            # relative velocity
            _ang_tmp = g[a].angv * g[a].R + g[b].R * g[b].angv
            vx_ab = g[a].vx - g[b].vx - _ang_tmp * ny
            vy_ab = g[a].vy - g[b].vy - _ang_tmp * nx

            vn = vx_ab * nx + vy_ab * ny
            vn_x = vn * nx
            vn_y = vn * ny

            vt_x = vx_ab - vn_x
            vt_y = vy_ab - vn_y

            vt = np.sqrt(vt_x**2 + vt_y**2)
            tx = 0
            ty = 0
            if vt > 0:
                tx = vt_x / vt
                ty = vt_y / vt
            vrt = vx_ab * tx + vy_ab * ty

            g[a].tang_disp += vrt * dt
            g[b].tang_disp += vrt * dt

            # find contants
            r_eff = (g[a].R * g[b].R) / (g[a].R + g[b].R)
            m_eff = (g[a].m * g[b].m) / (g[a].m + g[b].m)
            tmp_1 = (1. - g[a].nu * g[a].nu) / g[a].E
            tmp_2 = (1. - g[b].nu * g[b].nu) / g[b].E
            E_eff = 1. / (tmp_1 + tmp_2)

            kn = 4. / 3. * np.sqrt(r_eff) * E_eff
            kt = 2. / 7. * kn

            _tmp = np.log(g[a].e)
            alpha_1 = _tmp * np.sqrt(5. / (_tmp**2 + np.pi**2))
            eta_tmp = alpha_1 * np.sqrt(m_eff * kn)
            eta_n = eta_tmp * dn**(1. / 4.)
            eta_t = 1./2. * eta_n

            fn = - (kn * dn**(3./2.) - eta_n * vn)
            ft = - (kt * g[a].tang_disp + eta_t * vrt)

            # coulombs criterion
            if abs(ft) > abs(g[a].mu * fn):
                ft = -g[a].mu * fn

            g[a].fx = fn * nx + ft * tx
            g[a].fy = fn * ny + ft * ty
            g[a].t = np.cross([g[a].R*nx, g[a].R*ny], [ft*tx, ft*ty])

            g[b].fx = -fn * nx - ft * tx
            g[b].fy = -fn * ny - ft * ty
            g[b].t = -np.cross([g[b].R*nx, g[b].R*ny], [ft*tx, ft*ty])

        else:
            g[a].tang_disp = 0
            g[b].tang_disp = 0


def interparticle_force_linear(g, a, b):
    if a > b:
        x_ab = g[b].x - g[a].x
        y_ab = g[b].y - g[a].y
        dist = np.sqrt(x_ab * x_ab + y_ab * y_ab)

        # size of overlap
        dn = (g[a].R + g[b].R) - dist

        if dn > 0:
            # normal vector
            nx = x_ab / dist
            ny = y_ab / dist

            # relative velocity
            _ang_tmp = g[a].angv * g[a].R + g[b].R * g[b].angv
            vx_ab = g[a].vx - g[b].vx - _ang_tmp * ny
            vy_ab = g[a].vy - g[b].vy - _ang_tmp * nx

            vn = vx_ab * nx + vy_ab * ny
            vn_x = vn * nx
            vn_y = vn * ny

            vt_x = vx_ab - vn_x
            vt_y = vy_ab - vn_y

            vt = np.sqrt(vt_x**2 + vt_y**2)
            tx = 0
            ty = 0
            if vt > 0:
                tx = vt_x / vt
                ty = vt_y / vt
            vrt = vx_ab * tx + vy_ab * ty

            g[a].tang_disp += vrt * dt
            g[b].tang_disp += vrt * dt

            # find contants
            r_eff = (g[a].R * g[b].R) / (g[a].R + g[b].R)
            m_eff = (g[a].m * g[b].m) / (g[a].m + g[b].m)
            tmp_1 = (1. - g[a].nu * g[a].nu) / g[a].E
            tmp_2 = (1. - g[b].nu * g[b].nu) / g[b].E
            E_eff = 1. / (tmp_1 + tmp_2)

            kn = 4. / 3. * np.sqrt(r_eff) * E_eff
            kt = 2. / 5. * kn

            _tmp = abs(np.log(g[a].e))
            _tmp = 2 * np.sqrt(m_eff * kn) * _tmp
            alpha_1 = np.sqrt(1. / (_tmp**2 + np.pi**2))
            eta_n = alpha_1 * _tmp
            eta_t = 1./2. * eta_n

            fn = - (kn * dn**(3./2.) - eta_n * vn)
            ft = - (kt * g[a].tang_disp + eta_t * vrt)

            # coulombs criterion
            if abs(ft) > abs(g[a].mu * fn):
                ft = -g[a].mu * fn

            g[a].fx = fn * nx + ft * tx
            g[a].fy = fn * ny + ft * ty
            g[a].t = np.cross([g[a].R*nx, g[a].R*ny], [ft*tx, ft*ty])

            g[b].fx = -fn * nx - ft * tx
            g[b].fy = -fn * ny - ft * ty
            g[b].t = -np.cross([g[b].R*nx, g[b].R*ny], [ft*tx, ft*ty])

        else:
            g[a].tang_disp = 0
            g[b].tang_disp = 0


def interparticle_force(g, a, b):
    if a > b:
        # particle centre coordinate component differences
        x_ab = g[a].x - g[b].x
        y_ab = g[a].y - g[b].y

        # Particle centre distance
        dist = np.sqrt(x_ab * x_ab + y_ab * y_ab)

        # size of overlap
        dn = (g[a].R + g[b].R) - dist
        if dn > 0:
            dist_eff = 1. / dist
            xn = x_ab * dist_eff
            yn = y_ab * dist_eff

            # compute the velocity of the contact
            vx_ab = g[a].vx - g[b].vx
            vy_ab = g[a].vy - g[b].vy

            vn = -(vx_ab * xn + vy_ab * yn)
            vt = vx_ab * -yn + vy_ab * xn + g[a].R * g[a].angv - g[b].R * g[
                b].angv

            # find contact parameters
            r_eff = (g[a].R * g[b].R) / (g[a].R + g[b].R)
            m_eff = (g[a].m * g[b].m) / (g[a].m + g[b].m)
            tmp_1 = (1. - g[a].nu * g[a].nu) / g[a].E
            tmp_2 = (1. - g[b].nu * g[b].nu) / g[b].E
            E_eff = 1. / (tmp_1 + tmp_2)

            kn = 4. / 3. * np.sqrt(r_eff) * E_eff
            # kt = 2. / 7. * kn

            _tmp = abs(np.log(g[a].e))
            alpha_1 = _tmp * np.sqrt(5. / (_tmp**2 + np.pi**2))
            eta_tmp = alpha_1 * np.sqrt(m_eff * kn)
            eta_n = eta_tmp * dn**(1. / 4.)
            eta_t = 1./2. * eta_n

            # Compute force in local axes
            fn = kn * dn**(3. / 2.) - eta_n * vn
            ft = -eta_t * vt

            # rotation
            if (fn < 0):
                fn = 0

            if ft < (-g[0].mu * fn):
                ft = -g[0].mu * fn

            if ft > (g[0].mu * fn):
                ft = g[0].mu * fn

            # calculate sum of forces on a and b in global coordinates
            g[a].fx += fn * xn - ft * yn
            g[a].fy += fn * yn + ft * xn
            g[a].t += g[a].R * ft

            g[b].fx -= fn * xn + ft * yn
            g[b].fy -= fn * yn - ft * xn
            g[b].t += -ft * g[b].R


def interact_grains(grains, method='nonlinear'):
    # loop through particle a
    for a in range(len(grains)):
        # Loop through particle b
        for b in range(len(grains)):
            if method == 'linear':
                interact_grains_linear(grains, a, b)
            else:
                interparticle_force_hamid(grains, a, b)


def update_acc(grains):
    for g in grains:
        g.ax = g.fx / g.m
        g.ay = g.fy / g.m
        g.anga = g.t / g.I


def apply_gravity(grains):
    for g in grains:
        g.ay += -grav


def correction(grains):
    for g in grains:
        if g.fixed is False:
            g.vx += 0.5 * dt * g.ax
            g.vy += 0.5 * dt * g.ay
            g.angv += 0.5 * dt * g.anga
