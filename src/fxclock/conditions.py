"""Test-time grid conditions (pre-registered in configs/prereg_n2.json).

Each condition maps a native all-measured window to a KnotSeries.  Random conditions use
a per-window generator seeded by (condition_seed, window_index), identical for every model.
"""

import numpy as np

from . import knots as K

CONDITIONS = {
    # lossless knot refinement (virtual knots on the FOH reconstruction)
    "native": ("lossless", lambda ks, rng: ks),
    "foh_m2": ("lossless", lambda ks, rng: K.refine_uniform(ks, 2, "foh")),
    "foh_m4": ("lossless", lambda ks, rng: K.refine_uniform(ks, 4, "foh")),
    "foh_rand": ("lossless", lambda ks, rng: K.refine_random(ks, ks.n - 1, "foh", rng)),
    # lossless for the right-hold (ZOH) reconstruction only
    "zoh_m2": ("lossless_zoh", lambda ks, rng: K.refine_uniform(ks, 2, "zoh")),
    # real downsampling (measured knots removed on a regular pattern; aliasing possible)
    "down2": ("downsample", lambda ks, rng: K.downsample(ks, 2)),
    "down4": ("downsample", lambda ks, rng: K.downsample(ks, 4)),
    # same reconstructed path as down2, re-knotted with virtual knots at the native times
    "down2_reknot": ("downsample+virtual", lambda ks, rng: K.add_virtual(K.downsample(ks, 2), ks.t, "foh")),
    # missing observations (measured knots removed irregularly)
    "drop30": ("missing", lambda ks, rng: K.drop_random(ks, 0.3, rng)),
    "drop50": ("missing", lambda ks, rng: K.drop_random(ks, 0.5, rng)),
    "drop70": ("missing", lambda ks, rng: K.drop_random(ks, 0.7, rng)),
    "drop50_reknot": ("missing+virtual", lambda ks, rng: K.add_virtual(K.drop_random(ks, 0.5, rng), ks.t, "foh")),
}

COND_SEED = {name: 1000 + i for i, name in enumerate(CONDITIONS)}
# drop50_reknot must use the SAME drop pattern as drop50
COND_SEED["drop50_reknot"] = COND_SEED["drop50"]


def apply(name, ks, window_index):
    kind, fn = CONDITIONS[name]
    rng = np.random.default_rng([COND_SEED[name], int(window_index)])
    return fn(ks, rng)
