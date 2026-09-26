"""Per-window preprocessing (numpy, float64) and batch collation for every model family.

This is the complete path from raw knots (t_i, x_i) to model inputs; its wall-clock is
part of the reported cost.  Output query rules (published):

* clock family  : outputs at the fixed clock tau_j = j * Delta_c, j = 1..J with
                  J = floor((T_end - t_start) / Delta_c); the output at tau_j depends only on
                  the reconstructed path X on [t_start, tau_j]; the sequence readout is the
                  uniform mean over j (a time average over the clock).
* per-observation family : outputs at the knots t_k > t_start; readout is the time-weighted
                  mean sum_k w_k y_k with w_k = (t_k - t_{k-1}) / sum_k (t_k - t_{k-1}).
* nrde family   : state after the last clock window.
"""

import numpy as np
import torch

from .knots import clock_times, evaluate, segments


def n_ticks(t_start, t_end, clock_dt):
    return int(np.floor((t_end - t_start) / clock_dt + 1e-9))


def logsig2_windows(seg, x_start):
    """Exact depth-2 log-signature of Y(t) = (t, X(t)) (FOH path) over every query window.

    seg: output of knots.segments(..., "foh", ...); x_start: X(t_start) (C,).
    Level 1: Y(q_j) - Y(q_{j-1}).  Level 2: Levy areas A_ab = 1/2 int (Y^a - Y^a_{q_{j-1}}) dY^b - (a<->b), a<b.
    For a piecewise-affine path A = 1/2 sum_l (P_l x D_l - D_l x P_l), P_l = Y(s_l) - Y(q_{j-1}); every term
    is unchanged when a segment is split (virtual knot), so the features are refinement-invariant.
    """
    d, xs, xe, win, q = seg["d"], seg["xs"], seg["xe"], seg["win"], seg["q"]
    J = q.shape[0]
    q_prev = np.concatenate([[seg["s"][0]], q[:-1]])                      # window starts (t_start first)
    x_prev = np.concatenate([x_start[None], seg["xq"][:-1]], axis=0)       # X at window starts
    s_time = seg["s"]
    D = np.concatenate([d[:, None], xe - xs], axis=1)                         # (S, 1+C)
    P = np.concatenate([(s_time - q_prev[win])[:, None], xs - x_prev[win]], axis=1)
    A = 0.5 * (P[:, :, None] * D[:, None, :] - D[:, :, None] * P[:, None, :])  # (S, D, D)
    iu = np.triu_indices(D.shape[1], k=1)
    lev1 = np.zeros((J, D.shape[1]))
    lev2 = np.zeros((J, iu[0].shape[0]))
    np.add.at(lev1, win, D)
    np.add.at(lev2, win, A[:, iu[0], iu[1]])
    return np.concatenate([lev1, lev2], axis=1)


def global_logsig2(loc, D, basepoint=None):
    """Depth-2 log-signature of Y on [t_start, q_j] from the window ones (Chen): level 1 adds;
    A_global(j) = sum_{w<=j} [A_w + 1/2 (P_{w-1} x Delta_w - Delta_w x P_{w-1})],  P_{w-1} = sum_{v<w} Delta_v.

    basepoint: optional (D,) increment of a segment prepended before t_start (basepoint augmentation, as in
    signatory's basepoint=True): the path starts at the origin and jumps to Y(t_start).  Without it the
    features are translation-invariant and lose absolute levels (e.g. gravity direction in HAR)."""
    lev1 = loc[:, :D]
    P0 = np.zeros((1, D)) if basepoint is None else basepoint[None]
    P = P0 + np.concatenate([np.zeros((1, D)), np.cumsum(lev1, axis=0)[:-1]], axis=0)
    iu = np.triu_indices(D, k=1)
    cross = 0.5 * (P[:, :, None] * lev1[:, None, :] - lev1[:, :, None] * P[:, None, :])[:, iu[0], iu[1]]
    return np.concatenate([P0 + np.cumsum(lev1, axis=0), np.cumsum(loc[:, D:] + cross, axis=0)], axis=1)


def prep_window(ks, spec, ctx):
    """Return a dict of numpy arrays for one window (see module docstring for query rules)."""
    t0, t_end, ndt = ctx["t_start"], ctx["t_end"], ctx["native_dt"]
    fam = spec["family"]
    out = {}
    if fam in ("clock", "nrde"):
        r = spec["clock_r"]
        J = n_ticks(t0, t_end, r * ndt)
        q = clock_times(J, r, ndt)
    else:
        q = ks.t[ks.t > t0]
        out["measured"] = ks.measured[ks.t > t0]
    out["q"] = q
    feats = []
    if spec.get("stem"):
        seg = segments(ks, q, "foh" if spec["stem"] == "bilin" else spec["stem"], t0)
        for k in ("d", "xs", "xe", "win", "rem", "qdt"):
            out[k] = seg[k]
        xq = seg["xq"]
    else:
        xq = evaluate(ks, q, "foh")
    if fam == "nrde":
        seg = segments(ks, q, "foh", t0)
        out["ls"] = logsig2_windows(seg, evaluate(ks, np.array([t0]), "foh")[0])
        out["x0"] = evaluate(ks, np.array([t0]), "foh")[0]
        return out
    if "logsig_local" in spec["feats"]:
        seg = segments(ks, q, "foh", t0)
        loc = logsig2_windows(seg, evaluate(ks, np.array([t0]), "foh")[0])
        feats.append(loc)
        if "logsig_global" in spec["feats"]:
            x0 = evaluate(ks, np.array([t0]), "foh")[0]
            feats.append(global_logsig2(loc, ks.x.shape[1] + 1, basepoint=np.concatenate([[0.0], x0])))
    if "point" in spec["feats"]:
        feats.append(xq)
    if "binmean" in spec["feats"]:
        seg = segments(ks, q, "foh", t0)
        acc = np.zeros((q.shape[0], ks.x.shape[1]))
        np.add.at(acc, seg["win"], seg["d"][:, None] * 0.5 * (seg["xs"] + seg["xe"]))
        feats.append(acc / seg["qdt"][:, None])
    if "patch" in spec["feats"]:
        r = spec["clock_r"]
        qf = clock_times(q.shape[0] * r, 1, ndt)
        feats.append(evaluate(ks, qf, "foh").reshape(q.shape[0], r * ks.x.shape[1]))
    out["feats"] = np.concatenate(feats, axis=1) if feats else np.zeros((q.shape[0], 0))
    dt = np.diff(np.concatenate([[t0], q]))
    out["dt"] = dt
    out["w"] = dt / dt.sum() if fam == "perobs" else np.full(q.shape[0], 1.0 / q.shape[0])
    return out


def collate(items, dtype=torch.float32):
    """Pad a list of prep_window dicts into tensors (padding is inert, see stem/ssm docstrings)."""
    B = len(items)
    L = max(it["q"].shape[0] for it in items)
    out = {}

    def pad(key, shape_tail, fill=0.0, np_dtype=np.float64, length=None):
        n = length or L
        arr = np.full((B, n) + shape_tail, fill, dtype=np_dtype)
        for i, it in enumerate(items):
            v = it[key]
            arr[i, : v.shape[0]] = v
        return arr

    if "ls" in items[0]:
        out["ls"] = torch.as_tensor(pad("ls", items[0]["ls"].shape[1:]), dtype=dtype)
        out["x0"] = torch.as_tensor(np.stack([it["x0"] for it in items]), dtype=dtype)
        return out
    out["feats"] = torch.as_tensor(pad("feats", items[0]["feats"].shape[1:]), dtype=dtype)
    out["dt"] = torch.as_tensor(pad("dt", ()), dtype=dtype)
    out["w"] = torch.as_tensor(pad("w", ()), dtype=dtype)
    out["q"] = torch.as_tensor(pad("q", ()), dtype=dtype)
    if "measured" in items[0]:
        out["measured"] = torch.as_tensor(pad("measured", (), fill=0.0), dtype=torch.bool)
    out["uniform"] = all(it["q"].shape[0] == L for it in items) and bool(
        np.allclose(np.concatenate([it["dt"] for it in items]), items[0]["dt"][0], rtol=0, atol=1e-12))
    if "d" in items[0]:
        S = max(it["d"].shape[0] for it in items)
        C = items[0]["xs"].shape[1]
        out["d"] = torch.as_tensor(pad("d", (), length=S), dtype=dtype)
        out["xs"] = torch.as_tensor(pad("xs", (C,), length=S), dtype=dtype)
        out["xe"] = torch.as_tensor(pad("xe", (C,), length=S), dtype=dtype)
        out["rem"] = torch.as_tensor(pad("rem", (), length=S), dtype=dtype)
        out["win"] = torch.as_tensor(pad("win", (), fill=L, np_dtype=np.int64, length=S), dtype=torch.long)
        out["qdt"] = torch.as_tensor(pad("qdt", ()), dtype=dtype)
    return out
