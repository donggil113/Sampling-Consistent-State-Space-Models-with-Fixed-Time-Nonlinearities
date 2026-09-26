"""Diagonal linear recurrences h_k = a_k * h_{k-1} + b_k (h_{-1} = 0) along dim 1."""

import torch


def _shift(x, s, fill):
    pad = torch.full_like(x[:, :s], fill)
    return torch.cat([pad, x[:, :-s]], dim=1)


def diag_scan(a, b):
    """Inclusive Hillis-Steele scan; a, b: (B, L, ...) complex.  O(L log L) work, log2(L) vectorised steps.

    Only products of the a's appear (no division), so decaying factors underflow to 0 harmlessly.
    """
    L = a.shape[1]
    A, S = a, b
    s = 1
    while s < L:
        S = A * _shift(S, s, 0.0) + S
        A = A * _shift(A, s, 1.0)
        s *= 2
    return S


def diag_scan_seq(a, b):
    """Sequential reference (used only in tests)."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for k in range(a.shape[1]):
        h = a[:, k] * h + b[:, k]
        out.append(h)
    return torch.stack(out, dim=1)
