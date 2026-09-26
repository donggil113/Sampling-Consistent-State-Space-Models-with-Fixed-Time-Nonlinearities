"""Readouts, discrepancy metrics and small statistics helpers."""

import math

READOUTS = ("last", "sample_sum", "sample_mean", "time_riemann", "time_trapz", "time_exact")
SAMPLE_COUNT_READOUTS = ("sample_sum", "sample_mean")
TIME_WEIGHTED_READOUTS = ("time_riemann", "time_trapz", "time_exact")


def readouts(times, ys, integral=None):
    """Aggregate an output sequence y_0..y_K observed at t_0..t_K (y_0 = 0)."""
    K = len(ys) - 1
    T = times[-1] - times[0]
    obs = ys[1:]
    dts = [times[k] - times[k - 1] for k in range(1, K + 1)]
    out = {
        "last": ys[-1],
        "sample_sum": sum(obs),
        "sample_mean": sum(obs) / K,
        "time_riemann": sum(y * d for y, d in zip(obs, dts)) / T,
        "time_trapz": sum(0.5 * (ys[k - 1] + ys[k]) * dts[k - 1] for k in range(1, K + 1)) / T,
        "time_exact": None if integral is None else integral / T,
    }
    return out


def at_index(seq, idx):
    return [seq[i] for i in idx]


def l2(a):
    return math.sqrt(sum(abs(x) ** 2 for x in a))


def diff_l2(a, b):
    return math.sqrt(sum(abs(x - y) ** 2 for x, y in zip(a, b)))


def rel_l2(a, b):
    return diff_l2(a, b) / max(l2(b), 1e-300)


def flat_states(hs):
    return [x for h in hs for x in h]


def rms(a):
    return l2(a) / math.sqrt(max(len(a), 1))


def loglog_slope(xs, ys, floor=1e-300):
    """Least-squares slope of log(y) vs log(x); None if any y <= floor."""
    if len(xs) < 2 or any((y is None) or (y <= floor) for y in ys):
        return None
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    return sxy / sxx


def median(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def summarize(v):
    vals = [x for x in v if x is not None]
    return {
        "n": len(v),
        "n_valid": len(vals),
        "median": median(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }
