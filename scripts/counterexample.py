"""Minimal counterexamples for per-observation nonlinear updates (float64, numpy + fractions only).

Independent of src/fxclock (own formulas), so it also serves as a cross-check.

CE1  analytic: x(t) = t on [0, 1]; knots {0,1} vs {0,1/2,1} (virtual knot on the FOH line).
     per-knot sigma(u) = u^2, then exact integral of the re-reconstructed intermediate signal.
CE2  S5/FlowState-style 2-layer stack: exact ZOH layers with physical dt, GELU at every knot,
     right-hold re-reconstruction of the intermediate signal.  Grid change: split every held interval
     into m pieces with the SAME held value (lossless for the right-hold rule).  Compared with
     (a) the same stack with sigma = identity, (b) exact joint integration of the linear cascade,
     (c) the fixed-clock variant (sigma only at clock ticks, clock independent of the knots).
CE3  single selective layer (Mamba-like gate g(u) evaluated per knot, Delta = dt g(u), exact ZOH):
     invariant under ZOH splitting (reproduces N1 H1) but NOT under FOH virtual knots.
"""

import json
import math
import os
import sys
from fractions import Fraction as Fr

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/counterexample.json"


def ce1():
    sq = lambda u: u * u
    def trap(ts, vs):  # exact integral of the FOH reconstruction
        return sum((ts[i + 1] - ts[i]) * (vs[i] + vs[i + 1]) / 2 for i in range(len(ts) - 1))
    def hold(ts, vs):  # exact integral of the right-hold reconstruction
        return sum((ts[i + 1] - ts[i]) * vs[i + 1] for i in range(len(ts) - 1))
    G1 = [Fr(0), Fr(1)]
    G2 = [Fr(0), Fr(1, 2), Fr(1)]
    x = lambda t: t
    res = {
        "foh_knots_0_1": trap(G1, [sq(x(t)) for t in G1]),
        "foh_knots_0_half_1": trap(G2, [sq(x(t)) for t in G2]),
        "zoh_knots_0_1": hold(G1, [sq(x(t)) for t in G1]),
        "zoh_knots_0_half_1": hold(G2, [sq(x(t)) for t in G2]),
        "continuous_limit": Fr(1, 3),
        # fixed clock {0, 1/2, 1}: sigma only at clock ticks; X(tau) is identical for both knot sets
        "clock_foh_any_knots": trap([Fr(0), Fr(1, 2), Fr(1)], [sq(x(t)) for t in (Fr(0), Fr(1, 2), Fr(1))]),
        # affine sigma: invariant (Proposition in docs/THEORY.md)
        "affine_foh_knots_0_1": trap(G1, [3 * x(t) + 1 for t in G1]),
        "affine_foh_knots_0_half_1": trap(G2, [3 * x(t) + 1 for t in G2]),
    }
    return {k: str(v) for k, v in res.items()}


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def phi1(z):
    z = complex(z)
    if abs(z) < 1e-3:
        return 1 + z / 2 + z * z / 6 + z ** 3 / 24
    return (np.exp(z) - 1) / z


def zoh_layer(ts, u_held, lam, b, c, dvec):
    """Exact ZOH diag layer on right-held input; returns outputs y at ts[1:] (and state)."""
    h = np.zeros(len(lam), dtype=complex)
    ys = []
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        u = u_held[k - 1]
        h = np.exp(lam * dt) * h + dt * np.array([phi1(l * dt) for l in lam]) * b * u
        ys.append((c * h).sum().real + dvec * u)
    return np.array(ys)


def base_setup(seed=0, K=16, T=8.0):
    rng = np.random.default_rng(seed)
    lam1 = -np.exp(rng.uniform(np.log(0.2), np.log(2.0), 4)) + 1j * rng.uniform(0, 3, 4)
    lam2 = -np.exp(rng.uniform(np.log(0.2), np.log(2.0), 4)) + 1j * rng.uniform(0, 3, 4)
    p = [rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(4)]
    amps, oms, phs = rng.uniform(0.3, 1.0, 3), rng.uniform(0.5, 3.0, 3), rng.uniform(0, 2 * np.pi, 3)
    path = lambda t: float(np.sum(amps * np.sin(oms * t + phs)))
    tb = np.linspace(0.0, T, K + 1)
    return lam1, lam2, p, path, tb


def split(tb, vals, m):
    ts, vs, base_idx = [tb[0]], [], [0]
    for k in range(1, len(tb)):
        for j in range(1, m + 1):
            ts.append(tb[k] if j == m else tb[k - 1] + (tb[k] - tb[k - 1]) * j / m)
            vs.append(vals[k - 1])
        base_idx.append(len(ts) - 1)
    return np.array(ts), np.array(vs), np.array(base_idx)


def ce2(seed):
    lam1, lam2, (b1, c1, b2, c2), path, tb = base_setup(seed)
    u = np.array([path(t) for t in tb[1:]])
    out = {}
    ref = {}
    for m in (1, 2, 4, 8, 16):
        ts, vs, bi = split(tb, u, m)
        y1 = zoh_layer(ts, vs, lam1, b1, c1, 0.3)                     # layer 1 at every knot
        res = {"layer1_at_base": y1[bi[1:] - 1]}
        for name, sig in (("gelu", gelu), ("identity", lambda z: z)):
            y2 = zoh_layer(ts, sig(y1), lam2, b2, c2, 0.0)             # per-knot sigma, right-hold re-reconstruction
            res[f"perobs_{name}"] = y2[bi[1:] - 1]
        # fixed clock = base grid (does not move with the knots): sigma evaluated at clock ticks only
        y1_clock = y1[bi[1:] - 1]                                      # exact layer-1 output at clock ticks
        res["clock_gelu"] = zoh_layer(tb, gelu(y1_clock), lam2, b2, c2, 0.0)
        # exact joint integration of the linear cascade (sigma = identity), states stacked
        lam = np.concatenate([lam1, lam2])
        res["cascade_exact_identity"] = cascade_exact(ts, vs, lam1, lam2, b1, c1, b2, c2, 0.3)[bi[1:] - 1]
        if m == 1:
            ref = res
        out[m] = {k: float(np.linalg.norm(v - ref[k]) / max(np.linalg.norm(ref[k]), 1e-300)) for k, v in res.items()}
    return out


def cascade_exact(ts, vs, lam1, lam2, b1, c1, b2, c2, d1):
    """Exact ZOH of the joint linear system [h1; h2] driven by the held input (matrix exponential)."""
    n1, n2 = len(lam1), len(lam2)
    A = np.zeros((n1 + n2, n1 + n2), dtype=complex)
    A[:n1, :n1] = np.diag(lam1)
    A[n1:, n1:] = np.diag(lam2)
    # layer-2 input = y1(t) = Re(c1 . h1) + d1 u ; keep complex states but feed Re via conjugate pair trick:
    # write Re(c1.h1) = 0.5 (c1.h1 + conj(c1).conj(h1)); augment with conj(h1)
    Aa = np.zeros((2 * n1 + n2, 2 * n1 + n2), dtype=complex)
    Aa[:n1, :n1] = np.diag(lam1)
    Aa[n1:2 * n1, n1:2 * n1] = np.diag(np.conj(lam1))
    Aa[2 * n1:, 2 * n1:] = np.diag(lam2)
    Aa[2 * n1:, :n1] = 0.5 * b2[:, None] * c1[None, :]
    Aa[2 * n1:, n1:2 * n1] = 0.5 * b2[:, None] * np.conj(c1)[None, :]
    Ba = np.concatenate([b1, np.conj(b1), b2 * d1])
    from scipy.linalg import expm
    h = np.zeros(2 * n1 + n2, dtype=complex)
    ys = []
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        M = np.zeros((2 * n1 + n2 + 1, 2 * n1 + n2 + 1), dtype=complex)
        M[:-1, :-1] = Aa * dt
        M[:-1, -1] = Ba * dt * vs[k - 1]
        E = expm(M)
        h = E[:-1, :-1] @ h + E[:-1, -1]
        ys.append((c2 * h[2 * n1:]).sum().real)
    return np.array(ys)


def ce3(seed):
    lam1, _, (b1, c1, _, _), path, tb = base_setup(seed)
    w, beta = 1.2, 0.1
    g = lambda u: math.log1p(math.exp(w * u + beta))

    def selective(ts, vs):
        h = np.zeros(len(lam1), dtype=complex)
        ys = []
        for k in range(1, len(ts)):
            dt = ts[k] - ts[k - 1]
            uk = vs[k - 1]
            delta = dt * g(uk)
            h = np.exp(lam1 * delta) * h + delta * np.array([phi1(l * delta) for l in lam1]) * b1 * uk
            ys.append((c1 * h).sum().real)
        return np.array(ys)

    u = np.array([path(t) for t in tb[1:]])
    y_ref = selective(tb, u)
    out = {}
    for m in (2, 4, 8):
        ts, vs, bi = split(tb, u, m)                                     # ZOH-lossless split
        y_z = selective(ts, vs)[bi[1:] - 1]
        # FOH-lossless virtual knots: new values on the linear interpolant of the base samples
        uf = np.concatenate([[path(tb[0])], u])
        tsf = ts
        vsf = np.interp(tsf[1:], tb, uf)
        y_f = selective(tsf, vsf)[bi[1:] - 1]
        out[m] = {"zoh_split": float(np.linalg.norm(y_z - y_ref) / np.linalg.norm(y_ref)),
                  "foh_virtual": float(np.linalg.norm(y_f - y_ref) / np.linalg.norm(y_ref))}
    return out


def main():
    res = {"CE1": ce1(), "CE2": {str(s): ce2(s) for s in range(8)}, "CE3": {str(s): ce3(s) for s in range(8)}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["CE1"], indent=1))
    for key in ("layer1_at_base", "perobs_gelu", "perobs_identity", "clock_gelu", "cascade_exact_identity"):
        row = [np.median([res["CE2"][str(s)][m][key] for s in range(8)]) for m in (2, 4, 8, 16)]
        print(f"CE2 {key:24s} median rel change vs m=1 at m=2,4,8,16: " + " ".join(f"{v:.2e}" for v in row))
    for key in ("zoh_split", "foh_virtual"):
        row = [np.median([res["CE3"][str(s)][m][key] for s in range(8)]) for m in (2, 4, 8)]
        print(f"CE3 {key:12s} median rel change at m=2,4,8: " + " ".join(f"{v:.2e}" for v in row))


if __name__ == "__main__":
    main()
