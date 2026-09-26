"""FS1: does the published FlowState model change its forecast when only virtual knots are added?

Real series: ETTm1, column OT, 15-min sampling (FlowState's own benchmark family).
For each context window (L native samples, ending at the same physical time):
  native      : x, scale_factor s = get_fixed_factor("15T") = 0.25, prediction_length H
  refine r    : FOH virtual knots -> r(L-1)+1 samples at 15/r min (same reconstructed path), s/r, H*r
  down r      : every r-th measured sample (ending at the last one), s*r, H/r   (lossy; aliasing possible)
  zohR r      : every sample repeated r times (lossless for FlowState's right-hold kernel x_k = Abar x_{k-1} + Bbar u_k,
                including the implicit one-step hold of the first sample), s/r, H*r
Forecasts are compared at common physical future times.  A path-functional (sampling-consistent) model
would give identical forecasts under refinement (up to float32 rounding, ~1e-6).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get("FS_THREADS", "1")))
from tsfm_public import FlowStateForPrediction  # noqa: E402
from tsfm_public.models.flowstate.utils.utils import get_fixed_factor  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/flowstate_probe.json"
L, H, N_WIN = 1024, 96, 16
REVISIONS = {"v1.0 (main)": None, "r1.1": "r1.1"}


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def run(model, series, s, h):
    x = torch.tensor(series, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        out = model(past_values=x, scale_factor=float(s), prediction_length=int(h), batch_first=True,
                    prediction_type="median", output_hidden_states=True)
    fc = out.prediction_outputs[0, :, 0].numpy().astype(np.float64)
    bb = out.backbone_hidden_state[0, 0].numpy().astype(np.float64)
    return fc, bb


def main():
    df = pd.read_csv("/home/user/data/ETTm1.csv")
    y = df["OT"].to_numpy(np.float64)
    n = y.shape[0]
    test_start = 12 * 30 * 96 + 4 * 30 * 96          # standard ETTm1 split: 12/4/4 months
    ends = np.linspace(test_start + L, n - H - 1, N_WIN).astype(int)
    s = get_fixed_factor("15T")
    res = {"series": "ETTm1/OT", "L": L, "H": H, "scale_factor": s, "window_ends": ends.tolist(), "revisions": {}}
    for rev_name, rev in REVISIONS.items():
        kw = {"revision": rev} if rev else {}
        model = FlowStateForPrediction.from_pretrained("ibm-granite/granite-timeseries-flowstate-r1", **kw).eval()
        rows = []
        for e in ends:
            ctx = y[e - L + 1:e + 1]
            truth = y[e + 1:e + 1 + H]
            t0 = time.perf_counter()
            fc0, bb0 = run(model, ctx, s, H)
            fc0b, bb0b = run(model, ctx, s, H)                 # determinism check
            row = {"end": int(e), "repeat_rel_change": rel(fc0b, fc0), "mae_native": float(np.abs(fc0 - truth).mean())}
            for r in (2, 4):
                tn = np.arange(L, dtype=np.float64)
                tf = np.arange(r * (L - 1) + 1, dtype=np.float64) / r
                # FOH virtual knots (lossless for FOH); r-1 leading copies of x_0 keep FlowState's implicit
                # pre-window hold of the first sample at one native step (its kernel holds u_0 over one step)
                ref_ctx = np.concatenate([np.repeat(ctx[:1], r - 1), np.interp(tf, tn, ctx)])
                fcr, bbr = run(model, ref_ctx, s / r, H * r)
                fcr_c = fcr[r - 1::r][:H]
                row[f"refine{r}_rel_forecast_change"] = rel(fcr_c, fc0)
                row[f"refine{r}_rel_state_change"] = rel(bbr, bb0)
                row[f"refine{r}_mae"] = float(np.abs(fcr_c - truth).mean())
                # lossless for FlowState's own right-hold ZOH (first S5 layer exact by its App. B.1)
                for hold, seq in (("zohR", np.repeat(ctx, r)),):
                    fch, bbh = run(model, seq, s / r, H * r)
                    fch_c = fch[r - 1::r][:H]
                    row[f"{hold}{r}_rel_forecast_change"] = rel(fch_c, fc0)
                    row[f"{hold}{r}_rel_state_change"] = rel(bbh, bb0)
                dctx = ctx[::-1][::r][::-1]                       # every r-th sample, ending at the last one
                fcd, bbd = run(model, dctx, s * r, H // r)
                common = fc0[r - 1::r][: H // r]
                row[f"down{r}_rel_forecast_change"] = rel(fcd, common)
                row[f"down{r}_rel_state_change"] = rel(bbd, bb0)
                row[f"down{r}_mae_common"] = float(np.abs(fcd - truth[r - 1::r][: H // r]).mean())
                row[f"native_mae_common{r}"] = float(np.abs(common - truth[r - 1::r][: H // r]).mean())
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
            print(rev_name, e, {k: round(v, 4) for k, v in row.items() if "rel" in k})
        res["revisions"][rev_name] = {"n_params": sum(p.numel() for p in model.parameters()), "rows": rows}
    json.dump(res, open(OUT, "w"), indent=1)
    for rev_name, rr in res["revisions"].items():
        rows = rr["rows"]
        print(f"== {rev_name} ({rr['n_params']} params)")
        for k in rows[0]:
            if k not in ("end",):
                v = np.array([r[k] for r in rows])
                print(f"  {k:32s} median {np.median(v):.4g}  min {v.min():.4g}  max {v.max():.4g}")


if __name__ == "__main__":
    main()
