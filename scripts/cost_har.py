"""Cost including preprocessing, measured on an otherwise idle machine with 1 thread (pre-registered protocol).

For every inference rule and condition in {native, foh_m4, down2, drop50}, on the first 512 test windows:
  prep  : raw knots (after the grid change) -> model inputs, seconds per window (median of 3 repeats)
  fwd   : model forward, batch 64 (per-observation rules: 32, as in evaluation), seconds per window (median of 3)
Weights are the trained seed-0 checkpoints (timing does not depend on weight values, but this keeps shapes exact).
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fxclock.data_har import NATIVE_DT, load  # noqa: E402
from fxclock.models import SPECS, Net  # noqa: E402
from fxclock.train import predict, prep_all  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from run_har import GROUPS  # noqa: E402

torch.set_num_threads(1)
N_WIN, REPEATS = 512, 3
CONDS = ["native", "foh_m4", "down2", "drop50"]


def main():
    cfg_all = json.load(open("configs/prereg_n2.json"))
    cfg = dict(cfg_all["model"])
    cfg.update(cfg_all["training"])
    d = load(cfg_all["data"]["har_root"], cfg_all["data"]["har_cache"])
    mu, sd = d["Xtr"].mean((0, 1)), d["Xtr"].std((0, 1))
    X = (d["Xte"][:N_WIN] - mu) / sd
    ctx = {"t_start": 0.0, "t_end": 127 * NATIVE_DT, "native_dt": NATIVE_DT}
    out = {"n_windows": N_WIN, "repeats": REPEATS, "threads": 1, "rows": []}
    for g, specs in GROUPS.items():
        ck = f"results/raw/har/{g}__seed0.pt"
        state = torch.load(ck) if os.path.exists(ck) else None
        for name in specs:
            spec = SPECS[name]
            ls_scale = state["core.ls_scale"].numpy() if state is not None and "core.ls_scale" in state else (
                state["feat_scale"].numpy() if state is not None and "feat_scale" in state else None)
            if ls_scale is None and (spec["family"] == "nrde" or any(f.startswith("logsig") for f in spec["feats"])):
                ls_scale = np.ones(1)
            net = Net(spec, 9, 6, cfg, seed=0, native_dt=NATIVE_DT, ls_scale=ls_scale)
            if state is not None:
                net.load_state_dict(state)
            n_params = sum(p.numel() for p in net.parameters())
            bs = cfg_all["eval"]["batch_size_perobs"] if spec["family"] == "perobs" else 64
            for c in CONDS:
                tp, tf = [], []
                for _ in range(REPEATS):
                    items, t_prep = prep_all(X, spec, ctx, c)
                    t0 = time.perf_counter()
                    predict(net, items, bs, force_scan=False)
                    tf.append(time.perf_counter() - t0)
                    tp.append(t_prep)
                row = {"model": name, "group": g, "condition": c, "n_params": n_params,
                       "mean_len": float(np.mean([it["q"].shape[0] for it in items])),
                       "prep_us_per_window": 1e6 * float(np.median(tp)) / N_WIN,
                       "fwd_us_per_window": 1e6 * float(np.median(tf)) / N_WIN}
                row["total_us_per_window"] = row["prep_us_per_window"] + row["fwd_us_per_window"]
                out["rows"].append(row)
                print(f"{name:18s} {c:8s} len {row['mean_len']:6.1f} prep {row['prep_us_per_window']:8.1f} us "
                      f"fwd {row['fwd_us_per_window']:8.1f} us total {row['total_us_per_window']:8.1f} us  params {n_params}", flush=True)
    json.dump(out, open("results/cost_har.json", "w"), indent=1)


if __name__ == "__main__":
    main()
