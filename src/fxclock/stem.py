"""Exact linear input-integration stem.

Continuous-time diagonal complex SSM driven by the reconstructed path X(t) in R^C:

    dh/dt = Lambda h + B X(t),    h(t_start) = 0,    Lambda = diag(lambda_n), Re lambda_n < 0.

On a segment [s, e] of length d on which X is affine (X(s + r) = X(s) + r (X(e) - X(s)) / d),
variation of constants gives the exact update (first-order hold, FOH):

    h(e) = e^{Lambda d} h(s) + d [ (phi1 - phi2)(Lambda d) B X(s) + phi2(Lambda d) B X(e) ].

With X(s) = X(e) this is exact ZOH (d phi1 B u).  Contributions of all segments inside a
query window (q_{j-1}, q_j] are propagated DIRECTLY to q_j by e^{Lambda (q_j - e)} and summed,
and only the window states are composed sequentially:

    h(q_j) = e^{Lambda (q_j - q_{j-1})} h(q_{j-1}) + sum_{segments in window j} e^{Lambda (q_j - e)} c_seg.

Inserting a virtual knot on an affine piece splits one segment into two whose contributions
sum to the original one (semigroup property), so h(q_j) is unchanged in exact arithmetic.
"""

import math

import torch
from torch import nn

from .phi import phi12
from .scan import diag_scan


class ExactStem(nn.Module):
    """Parameters: lambda_n = -exp(log_decay_n) + i omega_n (1/s), complex B (N, C)."""

    def __init__(self, c_in, n_modes, tau_min, tau_max, f_max, seed=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        u = torch.rand(n_modes, generator=g, dtype=torch.float64)
        tau = torch.exp(math.log(tau_min) + u * (math.log(tau_max) - math.log(tau_min)))
        self.log_decay = nn.Parameter((-torch.log(tau)).float())
        self.omega = nn.Parameter((2 * math.pi * f_max * torch.rand(n_modes, generator=g, dtype=torch.float64)).float())
        s = (0.5 / c_in) ** 0.5
        self.B_re = nn.Parameter((s * torch.randn(n_modes, c_in, generator=g, dtype=torch.float64)).float())
        self.B_im = nn.Parameter((s * torch.randn(n_modes, c_in, generator=g, dtype=torch.float64)).float())
        self.n_modes = n_modes

    def lam(self):
        return torch.complex(-torch.exp(self.log_decay), self.omega)

    def forward(self, d, xs, xe, win, rem, qdt):
        """d, rem: (Bt, S); xs, xe: (Bt, S, C); win: (Bt, S) long in [0, J] (J = padding slot); qdt: (Bt, J).

        Returns h at the queries: (Bt, J, N) complex.
        """
        lam = self.lam()
        B = torch.complex(self.B_re, self.B_im)
        cdt = B.dtype
        z = d.unsqueeze(-1).to(cdt) * lam                       # (Bt, S, N)
        p1, p2 = phi12(z)
        bxs = xs.to(cdt) @ B.T                                  # (Bt, S, N)
        bxe = xe.to(cdt) @ B.T
        c = d.unsqueeze(-1).to(cdt) * ((p1 - p2) * bxs + p2 * bxe)
        c = c * torch.exp(rem.unsqueeze(-1).to(cdt) * lam)
        Bt, J = qdt.shape
        g = torch.zeros(Bt, J + 1, self.n_modes, dtype=cdt, device=c.device)
        g = g.scatter_add(1, win.unsqueeze(-1).expand(-1, -1, self.n_modes), c)[:, :J]
        a = torch.exp(qdt.unsqueeze(-1).to(cdt) * lam)         # (Bt, J, N)
        return diag_scan(a, g)


class BilinearStem(ExactStem):
    """Same parameterisation, but bilinear (Tustin) steps on the merged sample grid, S4/HiPPO style:

        h_k = (1 + z/2)/(1 - z/2) h_{k-1} + d/(1 - z/2) B x_k,   z = lambda d,

    as in the multirate PDM-speech SSM front end (Boulanger & Wood 2026, bilinear on raw samples, then
    decimation).  Not exact for the FOH path, so virtual knots change it (O(d^2) for smooth inputs).
    """

    def forward(self, d, xs, xe, win, rem, qdt):
        lam = self.lam()
        B = torch.complex(self.B_re, self.B_im)
        cdt = B.dtype
        z = d.unsqueeze(-1).to(cdt) * lam
        den = 1.0 - 0.5 * z
        a = (1.0 + 0.5 * z) / den
        b = d.unsqueeze(-1).to(cdt) / den * (xe.to(cdt) @ B.T)
        h = diag_scan(a, b)                                            # state at every segment end
        J = qdt.shape[1]
        last = (win.unsqueeze(-1) <= torch.arange(J, device=win.device)).sum(1) - 1   # (Bt, J) last segment of window j
        return torch.gather(h, 1, last.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.n_modes))


def stem_features(h):
    """Real features [Re h, Im h]."""
    return torch.cat([h.real, h.imag], dim=-1)
