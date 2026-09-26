"""Stable phi-functions of exponential integrators for complex tensors.

    phi1(z) = (e^z - 1) / z,         phi1(0) = 1
    phi2(z) = (e^z - 1 - z) / z^2,   phi2(0) = 1/2

Direct formulas lose precision for small |z| (phi2 loses about eps/|z|^2
relative accuracy).  For |z| < SERIES_R a truncated Taylor series in Horner
form is used instead.  Both branches are evaluated on "safe" arguments so
that the unused branch never produces inf/nan, which keeps autograd finite
(torch.where propagates nan gradients from the unused branch otherwise).
"""

import math

import torch

SERIES_R = 0.5
N_TERMS = 18  # truncation error < 0.5**19 / 20! ~ 8e-25


def cexpm1(z):
    """exp(z) - 1 for complex z without cancellation (same formula as n1ref.model.cexpm1)."""
    x, y = z.real, z.imag
    s = torch.sin(0.5 * y)
    re = torch.expm1(x) * torch.cos(y) - 2.0 * s * s
    im = torch.exp(x) * torch.sin(y)
    return torch.complex(re, im)


def _series(z, offset):
    """sum_{k=0}^{N_TERMS} z^k / (k + offset)!  (Horner)."""
    acc = torch.full_like(z, 1.0 / math.factorial(N_TERMS + offset))
    for k in range(N_TERMS - 1, -1, -1):
        acc = acc * z + 1.0 / math.factorial(k + offset)
    return acc


def phi12(z, series_r=SERIES_R):
    """Return (phi1(z), phi2(z)).  z: complex tensor (Re z <= 0 expected)."""
    small = z.abs() < series_r
    one = torch.ones_like(z)
    zs = torch.where(small, z, torch.zeros_like(z))
    zb = torch.where(small, one, z)
    p1_big = cexpm1(zb) / zb
    p2_big = (p1_big - 1.0) / zb
    p1 = torch.where(small, _series(zs, 1), p1_big)
    p2 = torch.where(small, _series(zs, 2), p2_big)
    return p1, p2


def phi12_naive(z):
    """Direct formulas without the series branch (numerics logs only)."""
    e = torch.exp(z)
    return (e - 1.0) / z, (e - 1.0 - z) / (z * z)
