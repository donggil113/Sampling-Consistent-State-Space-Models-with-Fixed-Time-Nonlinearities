"""Observation containers, published input-reconstruction rules, and grid changes.

Every knot carries a `measured` flag.  A *virtual knot* is a point placed ON the
published reconstruction of the current knots; it adds no information.  A
*measured knot* is a real sensor reading.  Three grid changes are kept apart:

  lossless knot refinement : add virtual knots (the reconstructed path is unchanged
                             by construction, for the rule used to place them)
  real downsampling        : delete measured knots on a regular pattern
  missing observations     : delete measured knots irregularly

Downsampling and missing observations change the reconstructed path.  Nothing
here recovers content lost to aliasing, and no function claims to.

Reconstruction rules (published; channels are reconstructed independently):

  "foh" : X(t) = linear interpolation of (t_i, x_i) for t in [t_0, t_{n-1}];
          X(t) = x_0 for t < t_0 and X(t) = x_{n-1} for t > t_{n-1} (constant hold).
  "zoh" : right hold, X(t) = x_i for t in (t_{i-1}, t_i]; X(t) = x_0 for t <= t_0;
          X(t) = x_{n-1} for t > t_{n-1}.  This is the hold implied by the usual
          per-step recurrence h_k = Abar(dt_k) h_{k-1} + Bbar(dt_k) u_k (S4/S5).
"""

from dataclasses import dataclass

import numpy as np

RULES = ("foh", "zoh")


@dataclass(frozen=True)
class KnotSeries:
    t: np.ndarray          # (n,) float64, strictly increasing
    x: np.ndarray          # (n, C) float64
    measured: np.ndarray   # (n,) bool

    def __post_init__(self):
        if self.t.ndim != 1 or self.x.shape[0] != self.t.shape[0] or self.measured.shape != self.t.shape:
            raise ValueError("shape mismatch")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("knot times must be strictly increasing")

    @property
    def n(self):
        return self.t.shape[0]


def from_uniform(x, dt, t0=0.0):
    """All-measured series on t_i = t0 + i*dt (times computed as i*dt, see clock_times)."""
    n = x.shape[0]
    t = t0 + np.arange(n, dtype=np.float64) * dt
    return KnotSeries(t, np.asarray(x, dtype=np.float64), np.ones(n, dtype=bool))


def evaluate(ks, q, rule):
    """Reconstructed path X(q) under `rule`; q: (J,) -> (J, C)."""
    q = np.asarray(q, dtype=np.float64)
    if rule == "foh":
        return np.stack([np.interp(q, ks.t, ks.x[:, c]) for c in range(ks.x.shape[1])], axis=1)
    if rule == "zoh":
        idx = np.clip(np.searchsorted(ks.t, q, side="left"), 0, ks.n - 1)
        return ks.x[idx]
    raise ValueError(rule)


def _insert(ks, new_t, new_x):
    keep = ~np.isin(new_t, ks.t)
    new_t, new_x = new_t[keep], new_x[keep]
    t = np.concatenate([ks.t, new_t])
    order = np.argsort(t, kind="stable")
    x = np.concatenate([ks.x, new_x])[order]
    m = np.concatenate([ks.measured, np.zeros(new_t.shape[0], dtype=bool)])[order]
    t = t[order]
    ok = np.concatenate([[True], np.diff(t) > 0])
    return KnotSeries(t[ok], x[ok], m[ok])


def add_virtual(ks, new_t, rule):
    """Insert virtual knots at times new_t, valued on the reconstruction `rule` (lossless for that rule)."""
    new_t = np.unique(np.asarray(new_t, dtype=np.float64))
    new_t = new_t[(new_t > ks.t[0]) & (new_t < ks.t[-1])]
    return _insert(ks, new_t, evaluate(ks, new_t, rule))


def refine_uniform(ks, m, rule):
    """m-1 equally spaced virtual knots inside every knot interval."""
    if m <= 1:
        return ks
    frac = np.arange(1, m, dtype=np.float64) / m
    new_t = (ks.t[:-1, None] + (ks.t[1:] - ks.t[:-1])[:, None] * frac[None, :]).ravel()
    return add_virtual(ks, new_t, rule)


def refine_random(ks, n_new, rule, rng):
    """n_new virtual knots at uniformly random times inside (t_0, t_{n-1})."""
    new_t = rng.uniform(ks.t[0], ks.t[-1], size=n_new)
    return add_virtual(ks, new_t, rule)


def downsample(ks, M):
    """Keep every M-th MEASURED knot (indices 0, M, 2M, ... among measured knots); drops virtual knots."""
    idx = np.flatnonzero(ks.measured)[::M]
    return KnotSeries(ks.t[idx], ks.x[idx], ks.measured[idx])


def drop_random(ks, p, rng):
    """Delete each interior MEASURED knot independently with prob p (first/last kept); drops virtual knots."""
    idx = np.flatnonzero(ks.measured)
    keep = rng.random(idx.shape[0]) >= p
    keep[0] = keep[-1] = True
    idx = idx[keep]
    return KnotSeries(ks.t[idx], ks.x[idx], ks.measured[idx])


def clock_times(n_ticks, ticks_per_native, native_dt):
    """Fixed physical-time clock tau_j = (j * r) * native_dt, j = 1..J.

    Written this way so that tau_j is bit-identical to the native knot time t_{j r} = (j r) * native_dt.
    The clock depends only on physical time, never on the observations.
    """
    j = np.arange(1, n_ticks + 1, dtype=np.float64)
    return (j * ticks_per_native) * native_dt


def segments(ks, q, rule, t_start):
    """Merge knots and query times into integration segments for the exact stem.

    Returns dict of numpy arrays (float64 unless noted):
      s   (S,)    segment start times; segments tile (t_start, q_J]
      d   (S,)    segment lengths
      xs  (S, C)  X at segment start (for "zoh": the held value on the segment)
      xe  (S, C)  X at segment end   (for "zoh": the held value on the segment)
      win (S,)    int, index j of the first query with q_j >= segment end
      rem (S,)    q_win - segment end  (>= 0)
      qdt (J,)    q_j - q_{j-1}, with q_0 = t_start
      xq  (J, C)  X(q_j) (point value of the reconstruction at the query)
    """
    q = np.asarray(q, dtype=np.float64)
    inner = ks.t[(ks.t > t_start) & (ks.t < q[-1])]
    P = np.union1d(inner, q)
    P = np.concatenate([[t_start], P[P > t_start]])
    s, e = P[:-1], P[1:]
    if rule == "foh":
        xs, xe = evaluate(ks, s, "foh"), evaluate(ks, e, "foh")
    elif rule == "zoh":
        xs = xe = evaluate(ks, e, "zoh")  # held value on (s, e] is the value of the first knot >= e
    else:
        raise ValueError(rule)
    win = np.searchsorted(q, e, side="left")
    rem = q[np.minimum(win, q.shape[0] - 1)] - e
    qdt = np.diff(np.concatenate([[t_start], q]))
    return {"q": q, "s": s, "d": e - s, "xs": xs, "xe": xe, "win": win, "rem": rem, "qdt": qdt, "xq": evaluate(ks, q, rule)}
