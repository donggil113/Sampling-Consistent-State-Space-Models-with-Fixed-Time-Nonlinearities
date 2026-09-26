"""FS3: FlowState encoder in float64 (weights cast), ZOH-lossless 2x refinement, fixed RevIN statistics.

Separates float32 rounding from the per-sample-nonlinearity effect: in exact arithmetic the first layer's
output at the last context step is invariant (exact ZOH + pointwise ops at that step); deeper layers are not.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
from tsfm_public import FlowStateForPrediction  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/flowstate_fp64_layers.json"
L = 1024


def encode(m, seq, s, mean, std, dtype):
    x = torch.tensor(seq, dtype=dtype).view(-1, 1, 1)                     # (L, B, 1)
    x = (x - mean) / std                                                  # fixed affine normalisation
    x = torch.cat([x, torch.zeros_like(x)], dim=-1)                       # [value, missing-mask]
    with torch.no_grad():
        e = m.model.embed(x)
        out = m.model.encoder(e, scale_factor=torch.ones(1, dtype=dtype) * s)
    return [h[-1, 0].double().numpy() for h in out.hidden_states]


def main():
    y = pd.read_csv("/home/user/data/ETTm1.csv")["OT"].to_numpy(np.float64)
    test_start = 12 * 30 * 96 + 4 * 30 * 96
    ends = np.linspace(test_start + L, y.shape[0] - 97, 16).astype(int)[:4]
    res = {}
    for rev_name, rev in {"v1.0 (main)": None, "r1.1": "r1.1"}.items():
        kw = {"revision": rev} if rev else {}
        rows = []
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            m = FlowStateForPrediction.from_pretrained("ibm-granite/granite-timeseries-flowstate-r1", **kw).eval().to(dtype)
            for e in ends:
                ctx = y[e - L + 1:e + 1]
                mean, std = float(ctx.mean()), float(ctx.std())
                h0 = encode(m, ctx, 0.25, mean, std, dtype)
                h1 = encode(m, np.repeat(ctx, 2), 0.125, mean, std, dtype)
                rows.append({"dtype": str(dtype), "end": int(e),
                             "per_layer_rel_change": [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(h1, h0)]})
                print(rev_name, rows[-1])
        torch.set_default_dtype(torch.float32)
        res[rev_name] = rows
    json.dump(res, open(OUT, "w"), indent=1)
    for rev_name, rows in res.items():
        for dt in ("torch.float32", "torch.float64"):
            v = np.array([r["per_layer_rel_change"] for r in rows if r["dtype"] == dt])
            print(rev_name, dt, "median per layer:", " ".join(f"{x:.2e}" for x in np.median(v, 0)))


if __name__ == "__main__":
    main()
