"""Time grids, input paths, interval splitting and new-observation refinement.

Two different grid changes are kept strictly separate:

* split_grid   : a base interval (t_{k-1}, t_k] is divided into m sub-intervals
                 and every sub-step receives the SAME held value u_k.  The
                 right-held physical input path is unchanged.
* refine_grid  : a base interval is divided into m sub-intervals and every new
                 sub-time s receives a NEW observation u(s) of the continuous
                 path.  The held path changes (new information).
"""

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SinePath:
    amps: tuple
    omegas: tuple
    phases: tuple

    def __call__(self, t):
        return sum(a * math.sin(w * t + p) for a, w, p in zip(self.amps, self.omegas, self.phases))

    @property
    def abs_bound(self):
        return sum(abs(a) for a in self.amps)


def sample_path(rng: random.Random, n_terms=3) -> SinePath:
    amps = tuple(rng.uniform(0.3, 1.0) for _ in range(n_terms))
    omegas = tuple(rng.uniform(0.5, 3.0) for _ in range(n_terms))
    phases = tuple(rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_terms))
    return SinePath(amps, omegas, phases)


@dataclass(frozen=True)
class Grid:
    times: tuple      # t_0 .. t_K
    values: tuple     # u_1 .. u_K
    base_index: tuple  # position in `times` of every base time t_0 .. t_Kb

    @property
    def n_steps(self):
        return len(self.values)

    @property
    def T(self):
        return self.times[-1] - self.times[0]


def base_times(T, K, kind="uniform", rng=None, jitter=0.4):
    dt = T / K
    ts = [k * dt for k in range(K + 1)]
    if kind == "uniform":
        return tuple(ts)
    if kind == "jittered":
        if rng is None:
            raise ValueError("jittered grid needs an rng")
        out = [0.0]
        for k in range(1, K):
            out.append(ts[k] + rng.uniform(-jitter, jitter) * dt)
        out.append(T)
        return tuple(out)
    raise ValueError(f"unknown grid kind {kind!r}")


def base_grid(times, path) -> Grid:
    values = tuple(path(t) for t in times[1:])
    return Grid(tuple(times), values, tuple(range(len(times))))


def _subdivide(times, values, m, mask, new_value):
    if m < 1:
        raise ValueError("m must be >= 1")
    K = len(values)
    if mask is None:
        mask = [True] * K
    if len(mask) != K:
        raise ValueError("mask must have one entry per base interval")
    out_t = [times[0]]
    out_v = []
    idx = [0]
    for k in range(1, K + 1):
        t0, t1 = times[k - 1], times[k]
        uk = values[k - 1]
        mk = m if mask[k - 1] else 1
        for j in range(1, mk + 1):
            s = t1 if j == mk else t0 + (t1 - t0) * j / mk
            out_t.append(s)
            out_v.append(uk if j == mk else new_value(s, uk))
        idx.append(len(out_t) - 1)
    return Grid(tuple(out_t), tuple(out_v), tuple(idx))


def split_grid(grid: Grid, m, mask=None) -> Grid:
    """Split intervals; every sub-step repeats the held base value (path unchanged)."""
    return _subdivide(grid.times, grid.values, m, mask, lambda s, uk: uk)


def refine_grid(grid: Grid, m, path, mask=None) -> Grid:
    """Insert new observations u(s) at the new sub-times (nested grids)."""
    return _subdivide(grid.times, grid.values, m, mask, lambda s, uk: path(s))


def first_half_mask(grid: Grid):
    half = grid.times[0] + 0.5 * grid.T
    return [grid.times[k] <= half + 1e-12 for k in range(1, len(grid.times))]


def held_value_at(grid: Grid, t):
    """Right-held path value at time t in (t_0, t_K]."""
    for k in range(1, len(grid.times)):
        if t <= grid.times[k]:
            return grid.values[k - 1]
    raise ValueError("t outside grid")
