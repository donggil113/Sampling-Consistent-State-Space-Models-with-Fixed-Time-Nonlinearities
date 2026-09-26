"""S4D-Lin style diagonal SSM layers in physical time, and the deep backbone.

One layer (per channel h, modes n; SISO per channel as in S4D):
    x' = lambda_{hn} x + u_h(t),   y_h = Re sum_n C_{hn} x_n + D_h u_h,
    lambda_{hn} = r_h (-exp(a_{hn}) + i b_{hn}),  r_h = exp(log_rate_h)  [1/s]
The layer input is taken as right-held (ZOH) between its sample times, so a step of
length dt is exact ZOH: xbar = e^{lambda dt} x + dt phi1(lambda dt) u.

* fixed clock   : dt = Delta_c for every step (LTI) -> FFT convolution.
* per observation: dt_k = t_k - t_{k-1} (physical step-size rescaling, S5/FlowState style)
                   -> time-varying scan; for a uniform grid it reduces to the LTI case.
In both cases the pointwise nonlinearity (GELU + GLU) is applied at every sample of
the sequence the layer runs on.  That sequence is the fixed clock for clocked models
and the observation knots for per-observation models.  This is the only difference.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .phi import cexpm1, phi12
from .scan import diag_scan


class DiagSSM(nn.Module):
    def __init__(self, H, N, dt_ref, dt_min=1e-3, dt_max=1e-1, seed=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        u = torch.rand(H, generator=g, dtype=torch.float64)
        log_dt = math.log(dt_min) + u * (math.log(dt_max) - math.log(dt_min))
        self.log_rate = nn.Parameter((log_dt - math.log(dt_ref)).float())       # r_h = Delta_h / dt_ref
        self.log_neg_re = nn.Parameter(torch.full((H, N), math.log(0.5), dtype=torch.float32))
        self.im = nn.Parameter((math.pi * torch.arange(N, dtype=torch.float32)).expand(H, N).clone())
        s = 0.5 ** 0.5
        self.C_re = nn.Parameter((s * torch.randn(H, N, generator=g, dtype=torch.float64)).float())
        self.C_im = nn.Parameter((s * torch.randn(H, N, generator=g, dtype=torch.float64)).float())
        self.D = nn.Parameter(torch.randn(H, generator=g, dtype=torch.float64).float())

    def lam(self):
        return torch.exp(self.log_rate)[:, None] * torch.complex(-torch.exp(self.log_neg_re), self.im)

    def forward(self, u, dt):
        """u: (Bt, L, H) real.  dt: python float (LTI) or tensor (Bt, L) of step lengths."""
        lam = self.lam()
        C = torch.complex(self.C_re, self.C_im).to(lam.dtype)
        if isinstance(dt, float):
            return self._lti(u, dt, lam, C)
        cdt = lam.dtype
        z = dt.to(u.dtype)[..., None, None].to(cdt) * lam           # (Bt, L, H, N)
        a = torch.exp(z)
        # dt * phi1(lambda dt) = expm1(z) / lambda: exact, no cancellation (lambda != 0), and no series
        # intermediates kept by autograd (the phi12 series made per-observation training OOM, deviation D5)
        b = cexpm1(z) / lam * u[..., None].to(cdt)
        x = diag_scan(a, b)
        y = (x * C).sum(-1).real
        return y + self.D * u

    def _lti(self, u, dt, lam, C):
        L = u.shape[1]
        z = lam * dt                                                   # (H, N)
        p1, _ = phi12(z)
        ell = torch.arange(L, dtype=u.dtype, device=u.device)
        K = ((C * dt * p1)[..., None] * torch.exp(z[..., None] * ell.to(z.dtype))).sum(1).real   # (H, L)
        n = 2 * L
        y = torch.fft.irfft(torch.fft.rfft(u.transpose(1, 2), n=n) * torch.fft.rfft(K, n=n), n=n)[..., :L]
        return y.transpose(1, 2) + self.D * u


class Block(nn.Module):
    def __init__(self, H, N, dt_ref, dropout, seed=None):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.ssm = DiagSSM(H, N, dt_ref, seed=seed)
        self.out = nn.Linear(H, 2 * H)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dt):
        y = self.ssm(self.norm(x), dt)
        y = F.glu(self.out(F.gelu(y)), dim=-1)
        return x + self.drop(y)


class SSMBackbone(nn.Module):
    def __init__(self, f_in, H, N, n_layers, dt_ref, dropout, n_out, seed=0):
        super().__init__()
        self.enc = nn.Linear(f_in, H)
        self.blocks = nn.ModuleList([Block(H, N, dt_ref, dropout, seed=seed * 1000 + i) for i in range(n_layers)])
        self.norm = nn.LayerNorm(H)
        self.head = nn.Linear(H, n_out)

    def forward(self, feats, dt, weights):
        """feats (Bt, L, F); dt float or (Bt, L); weights (Bt, L) pooling weights (sum to 1 per row)."""
        x = self.enc(feats)
        for b in self.blocks:
            x = b(x, dt)
        x = self.norm(x)
        pooled = (x * weights.unsqueeze(-1)).sum(1)
        return self.head(pooled)


class TransformerBackbone(nn.Module):
    """Encoder-only Transformer on a fixed clock; sinusoidal encoding of PHYSICAL time tau_j."""

    def __init__(self, f_in, H, n_layers, n_heads, dropout, n_out, time_scale):
        super().__init__()
        self.enc = nn.Linear(f_in, H)
        layer = nn.TransformerEncoderLayer(H, n_heads, 2 * H, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(H)
        self.head = nn.Linear(H, n_out)
        self.H = H
        self.time_scale = time_scale

    def time_encoding(self, t):
        half = self.H // 2
        freqs = torch.exp(-math.log(1e3) * torch.arange(half, dtype=t.dtype, device=t.device) / half) / self.time_scale
        ang = t[..., None] * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, feats, times, weights):
        x = self.enc(feats) + self.time_encoding(times)
        x = self.norm(self.tr(x))
        return self.head((x * weights.unsqueeze(-1)).sum(1))
