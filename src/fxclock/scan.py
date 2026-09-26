"""Diagonal linear recurrences h_k = a_k * h_{k-1} + b_k (h_{-1} = 0) along dim 1."""

import torch


def _shift(x, s, fill):
    pad = torch.full_like(x[:, :s], fill)
    return torch.cat([pad, x[:, :-s]], dim=1)


def _forward_loop(a, b):
    h = torch.zeros_like(b[:, 0])
    out = torch.empty_like(b)
    for k in range(a.shape[1]):
        h = a[:, k] * h + b[:, k]
        out[:, k] = h
    return out


class _DiagScan(torch.autograd.Function):
    """h_k = a_k h_{k-1} + b_k with a hand-written adjoint (stores only a and h; O(L) memory in the graph).

    Backward (PyTorch conjugate-Wirtinger convention, grad(a*h)/da = g * conj(h)):
        lam_k = g_k + conj(a_{k+1}) lam_{k+1},   grad_b_k = lam_k,   grad_a_k = lam_k * conj(h_{k-1}).
    """

    @staticmethod
    def forward(ctx, a, b):
        h = _forward_loop(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, g):
        a, h = ctx.saved_tensors
        L = a.shape[1]
        lam = torch.empty_like(g)
        acc = torch.zeros_like(g[:, 0])
        for k in range(L - 1, -1, -1):
            acc = g[:, k] + (torch.conj(a[:, k + 1]) * acc if k + 1 < L else 0.0)
            lam[:, k] = acc
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        return lam * torch.conj(h_prev), lam


def diag_scan(a, b):
    """Sequential scan (exact order of operations of the recurrence).  On CPU this is ~20x faster than the
    Hillis-Steele form for the sizes used here (memory traffic dominates), see diag_scan_parallel.
    With autograd, a custom adjoint keeps the graph at O(L) memory (see _DiagScan)."""
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return _DiagScan.apply(a, b)
    return _forward_loop(a, b)


def diag_scan_parallel(a, b):
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
