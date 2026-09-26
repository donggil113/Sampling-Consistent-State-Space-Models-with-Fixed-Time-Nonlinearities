"""Diagonal selective SSM toy in pure Python (float64 / complex128).

Continuous-time model (one scalar input channel, N diagonal modes):

    dh_n/dt = g(u(t)) * (lambda_n * h_n + B_n(u(t)) * u(t))
    y(t)    = Re( sum_n C_n(u(t)) * h_n(t) )

with a positive selection gate g(u) = softplus(w*u + beta) shared across modes
(as Delta is shared across the state dimension in Mamba-style layers).

Discrete recurrences use the right-endpoint hold: the step that ends at t_k
uses u_k = u(t_k) and dt_k = t_k - t_{k-1}.  A variant name is
"<scheme>_<dtmode>" with scheme in {zoh, eulerB, bilinear} and dtmode in
{dt, nodt}:

    dt   : Delta_k = dt_k * g(u_k)         (step size tied to physical time)
    nodt : Delta_k = tau  * g(u_k)         (step size ignores dt; grid-dependent)
"""

import cmath
import math
import random
from dataclasses import dataclass

SCHEMES = ("zoh", "eulerB", "bilinear")
DTMODES = ("dt", "nodt")
VARIANTS = tuple(f"{s}_{d}" for d in DTMODES for s in SCHEMES)

_SERIES_THRESHOLD = 0.1
_SERIES_TERMS = 12


def softplus(x):
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def cexpm1(z):
    """exp(z) - 1 for complex z without cancellation for small |z|."""
    z = complex(z)
    x, y = z.real, z.imag
    if y == 0.0:
        return complex(math.expm1(x), 0.0)
    s = math.sin(0.5 * y)
    re = math.expm1(x) * math.cos(y) - 2.0 * s * s
    im = math.exp(x) * math.sin(y)
    return complex(re, im)


def phi1(z):
    """(exp(z) - 1) / z, with phi1(0) = 1."""
    z = complex(z)
    if abs(z) < _SERIES_THRESHOLD:
        return 1.0 + phi1m1(z)
    return cexpm1(z) / z


def phi1m1(z):
    """phi1(z) - 1 = sum_{k>=1} z^k / (k+1)!, accurate for small |z|."""
    z = complex(z)
    if abs(z) < _SERIES_THRESHOLD:
        acc = 0.0 + 0.0j
        term = 1.0 + 0.0j
        for k in range(1, _SERIES_TERMS + 1):
            term = term * z / (k + 1)
            acc += term
        return acc
    return (cexpm1(z) - z) / z


@dataclass(frozen=True)
class ToyParams:
    lam: tuple  # complex, Re < 0
    w: float
    beta: float
    b0: tuple
    b1: tuple
    c0: tuple
    c1: tuple

    @property
    def n(self):
        return len(self.lam)

    def gate(self, u):
        return softplus(self.w * u + self.beta)

    def B(self, u):
        return [b0 + b1 * u for b0, b1 in zip(self.b0, self.b1)]

    def C(self, u):
        return [c0 + c1 * u for c0, c1 in zip(self.c0, self.c1)]

    def readout(self, u, h):
        return sum(c * hn for c, hn in zip(self.C(u), h)).real

    @property
    def lam_abs_max(self):
        return max(abs(l) for l in self.lam)


def sample_params(rng: random.Random, n=4, complex_modes=False) -> ToyParams:
    lam = []
    for _ in range(n):
        re = -math.exp(rng.uniform(math.log(0.2), math.log(2.0)))
        im = rng.uniform(0.0, 4.0) if complex_modes else 0.0
        lam.append(complex(re, im))

    def vec(sd):
        if complex_modes:
            s = sd / math.sqrt(2.0)
            return tuple(complex(rng.gauss(0.0, s), rng.gauss(0.0, s)) for _ in range(n))
        return tuple(complex(rng.gauss(0.0, sd), 0.0) for _ in range(n))

    w = rng.uniform(-1.5, 1.5)
    beta = rng.uniform(-0.5, 0.5)
    b0, b1, c0, c1 = vec(1.0), vec(0.5), vec(1.0), vec(0.5)
    return ToyParams(tuple(lam), w, beta, b0, b1, c0, c1)


def parse_variant(variant):
    scheme, dtmode = variant.split("_")
    if scheme not in SCHEMES or dtmode not in DTMODES:
        raise ValueError(f"unknown variant {variant!r}")
    return scheme, dtmode


def step_coeffs(params: ToyParams, u, dt, variant, tau, naive=False):
    """Return (abar, bbar, delta) for one step ending at an observation u.

    bbar multiplies the scalar input u, i.e. h <- abar*h + bbar*u.
    naive=True uses (exp(z)-1)/lambda instead of expm1 (for numerics logs only).
    """
    scheme, dtmode = parse_variant(variant)
    g = params.gate(u)
    delta = dt * g if dtmode == "dt" else tau * g
    B = params.B(u)
    abar, bbar = [], []
    for lam, b in zip(params.lam, B):
        z = delta * lam
        if scheme == "zoh":
            a = cmath.exp(z)
            em1 = (cmath.exp(z) - 1.0) if naive else cexpm1(z)
            bb = em1 / lam * b
        elif scheme == "eulerB":
            a = cmath.exp(z)
            bb = delta * b
        else:  # bilinear
            den = 1.0 - 0.5 * z
            a = (1.0 + 0.5 * z) / den
            bb = delta / den * b
        abar.append(a)
        bbar.append(bb)
    return abar, bbar, delta


def run_scan(params: ToyParams, times, values, variant, tau, naive=False):
    """Run the recurrence on a grid.

    times  : [t_0, ..., t_K]  (h(t_0) = 0)
    values : [u_1, ..., u_K]  (u_k is held on (t_{k-1}, t_k])
    Returns dict with
      h  : list of state tuples at t_0..t_K
      y  : outputs at t_0..t_K (y_0 = 0 because h_0 = 0)
      integral : exact integral of the intra-step continuous output under the
                 hold (zoh_* only; None otherwise)
    """
    if len(times) != len(values) + 1:
        raise ValueError("need len(times) == len(values) + 1")
    scheme, _ = parse_variant(variant)
    h = [0.0 + 0.0j] * params.n
    hs = [tuple(h)]
    ys = [0.0]
    integral = 0.0 if scheme == "zoh" else None
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        if dt <= 0.0:
            raise ValueError("times must be strictly increasing")
        u = values[k - 1]
        abar, bbar, delta = step_coeffs(params, u, dt, variant, tau, naive=naive)
        if integral is not None:
            integral += _zoh_step_integral(params, u, dt, delta, h)
        h = [a * hn + bb * u for a, bb, hn in zip(abar, bbar, h)]
        hs.append(tuple(h))
        ys.append(params.readout(u, h))
    return {"h": hs, "y": ys, "integral": integral}


def _zoh_step_integral(params, u, dt, delta, h0):
    """Integral over one held step of y(s) = Re(C(u) . h(s)).

    Within the step the zoh_* recurrence is the exact flow of
    dh/ds = (delta/dt) * (lambda*h + B(u)*u), s in [0, dt], so with z = delta*lambda
        int h = dt*phi1(z)*h0 + (dt/lambda)*(phi1(z)-1)*B(u)*u.
    """
    B = params.B(u)
    C = params.C(u)
    acc = 0.0 + 0.0j
    for lam, b, c, hn in zip(params.lam, B, C, h0):
        z = delta * lam
        ih = dt * phi1(z) * hn + (dt / lam) * phi1m1(z) * b * u
        acc += c * ih
    return acc.real
