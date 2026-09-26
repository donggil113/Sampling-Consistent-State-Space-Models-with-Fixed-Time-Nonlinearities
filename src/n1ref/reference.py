"""Independent continuous-time references (classical RK4), not using ZOH formulas.

ct_held : the continuous-time model driven by the right-held piecewise-constant
          path of a grid.  Within an interval the ODE is autonomous and linear,
          h' = alpha*h + beta, so classical RK4 with step d reduces exactly to
          h <- R(alpha*d)*h + d*P(alpha*d)*beta with the RK4 polynomials
          R(z)=1+z+z^2/2+z^3/6+z^4/24 and P(z)=1+z/2+z^2/6+z^3/24.
ct_true : the continuous-time model driven by the true continuous u(t),
          integrated with generic classical RK4 (time-varying coefficients).
Time integrals of y use composite Simpson on the RK4 nodes.
"""

import math


def _rk4_R(z):
    return 1.0 + z * (1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)))


def _rk4_P(z):
    return 1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0))


def _even_ceil(x, minimum):
    n = max(minimum, int(math.ceil(x)))
    return n + (n % 2)


def ct_held(params, grid, min_sub=64, zmax=0.002):
    """RK4 reference on the right-held path of `grid` (amendment A1 sub-steps)."""
    n = params.n
    h = [0.0 + 0.0j] * n
    hs = [tuple(h)]
    ys = [0.0]
    integral = 0.0
    total_sub = 0
    for k in range(1, len(grid.times)):
        dt = grid.times[k] - grid.times[k - 1]
        u = grid.values[k - 1]
        g = params.gate(u)
        nsub = _even_ceil(dt * g * params.lam_abs_max / zmax, min_sub)
        total_sub += nsub
        d = dt / nsub
        B = params.B(u)
        C = params.C(u)
        R = [_rk4_R(g * lam * d) for lam in params.lam]
        P = [_rk4_P(g * lam * d) for lam in params.lam]
        forcing = [g * b * u for b in B]
        # Simpson over the nsub+1 nodes of this interval
        def yval(hv):
            return sum(c * hn for c, hn in zip(C, hv)).real

        simpson = yval(h)
        for j in range(1, nsub + 1):
            h = [r * hn + d * p * f for r, p, f, hn in zip(R, P, forcing, h)]
            yj = yval(h)
            simpson += yj * (1.0 if j == nsub else (4.0 if j % 2 == 1 else 2.0))
        integral += simpson * d / 3.0
        hs.append(tuple(h))
        ys.append(yval(h))
    return {"h": hs, "y": ys, "integral": integral, "total_substeps": total_sub}


def ct_true(params, path, times, n_sub=512):
    """Generic RK4 on the continuous path; outputs at `times` (which include t_0)."""
    if n_sub % 2:
        raise ValueError("n_sub must be even for Simpson")
    n = params.n
    lam = params.lam

    def rhs(t, h):
        u = path(t)
        g = params.gate(u)
        B = params.B(u)
        return [g * (l * hn + b * u) for l, b, hn in zip(lam, B, h)]

    def yval(t, h):
        return params.readout(path(t), h)

    h = [0.0 + 0.0j] * n
    hs = [tuple(h)]
    ys = [0.0]
    integral = 0.0
    for k in range(1, len(times)):
        t0, t1 = times[k - 1], times[k]
        d = (t1 - t0) / n_sub
        simpson = yval(t0, h)
        t = t0
        for j in range(1, n_sub + 1):
            k1 = rhs(t, h)
            k2 = rhs(t + 0.5 * d, [hn + 0.5 * d * a for hn, a in zip(h, k1)])
            k3 = rhs(t + 0.5 * d, [hn + 0.5 * d * a for hn, a in zip(h, k2)])
            k4 = rhs(t + d, [hn + d * a for hn, a in zip(h, k3)])
            h = [hn + d / 6.0 * (a + 2.0 * b + 2.0 * c + e) for hn, a, b, c, e in zip(h, k1, k2, k3, k4)]
            t = t0 + j * d
            yj = yval(t, h)
            simpson += yj * (1.0 if j == n_sub else (4.0 if j % 2 == 1 else 2.0))
        integral += simpson * d / 3.0
        hs.append(tuple(h))
        ys.append(yval(t1, h))
    return {"h": hs, "y": ys, "integral": integral}
