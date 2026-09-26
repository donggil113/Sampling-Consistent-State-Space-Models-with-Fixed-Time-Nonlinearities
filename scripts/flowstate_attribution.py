"""FS2: where does FlowState's grid dependence come from?

Same ETTm1 windows as FS1 (first N_WIN of them).  Variant "fixed-RevIN": the causal RevIN statistics
(running mean/std over SAMPLE COUNTS) are replaced by one constant mean/std per context, taken from the
native context and used for every grid variant, i.e. an affine, time-invariant normalisation that commutes
with any reconstruction rule.  Under ZOH-lossless refinement the first S5 layer is then exact (FlowState
App. B.1), so any remaining change must come from per-sample operations feeding deeper layers.
Per-layer relative change of the encoder output at the LAST context step is recorded.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get("FS_THREADS", "1")))
from tsfm_public import FlowStateForPrediction  # noqa: E402
from tsfm_public.models.flowstate import modeling_flowstate as MF  # noqa: E402
from tsfm_public.models.flowstate.utils.utils import get_fixed_factor  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/flowstate_attribution.json"
L, H, N_WIN = 1024, 96, 8
FIXED = {"on": False, "mean": None, "std": None}
_orig_stats = MF.FlowStateCausalRevIN._get_statistics


def _fixed_stats(self, x):
    if not FIXED["on"]:
        return _orig_stats(self, x)
    shape = x.shape[:-1] + (1,)
    self.mean = torch.full(shape, FIXED["mean"], dtype=x.dtype)
    self.stdev = torch.full(shape, FIXED["std"], dtype=x.dtype)


MF.FlowStateCausalRevIN._get_statistics = _fixed_stats


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def run(model, series, s, h):
    x = torch.tensor(series, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        o = model(past_values=x, scale_factor=float(s), prediction_length=int(h), batch_first=True,
                  prediction_type="median", output_hidden_states=True)
    fc = o.prediction_outputs[0, :, 0].numpy().astype(np.float64)
    hs = o.hidden_states
    last = []
    for t in hs:
        a = t.detach().numpy()
        if a.ndim == 3 and a.shape[1] == 1:            # (seq, batch, dim): last real context step
            last.append(a[-1, 0].astype(np.float64))
    return fc, last


def main():
    y = pd.read_csv("/home/user/data/ETTm1.csv")["OT"].to_numpy(np.float64)
    n = y.shape[0]
    test_start = 12 * 30 * 96 + 4 * 30 * 96
    ends = np.linspace(test_start + L, n - H - 1, 16).astype(int)[:N_WIN]
    s = get_fixed_factor("15T")
    res = {"ends": ends.tolist(), "revisions": {}}
    for rev_name, rev in {"v1.0 (main)": None, "r1.1": "r1.1"}.items():
        kw = {"revision": rev} if rev else {}
        model = FlowStateForPrediction.from_pretrained("ibm-granite/granite-timeseries-flowstate-r1", **kw).eval()
        rows = []
        for e in ends:
            ctx = y[e - L + 1:e + 1]
            row = {"end": int(e)}
            for mode in ("causal_revin", "fixed_revin"):
                FIXED.update(on=(mode == "fixed_revin"), mean=float(ctx.mean()), std=float(ctx.std()))
                fc0, l0 = run(model, ctx, s, H)
                r = 2
                variants = {"zohR2": np.repeat(ctx, r),
                            "foh2": np.concatenate([np.repeat(ctx[:1], r - 1), np.interp(np.arange(r * (L - 1) + 1) / r, np.arange(L), ctx)])}
                for vn, seq in variants.items():
                    fc, lv = run(model, seq, s / r, H * r)
                    row[f"{mode}__{vn}__forecast"] = rel(fc[r - 1::r][:H], fc0)
                    row[f"{mode}__{vn}__per_layer_last_step"] = [rel(a, b) for a, b in zip(lv, l0)]
            FIXED["on"] = False
            rows.append(row)
            print(rev_name, e, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in row.items() if k != "end"})
        res["revisions"][rev_name] = rows
    json.dump(res, open(OUT, "w"), indent=1)
    for rev_name, rows in res["revisions"].items():
        print("==", rev_name)
        for k in rows[0]:
            if k == "end":
                continue
            v = np.array([r[k] for r in rows])
            if v.ndim == 1:
                print(f"  {k:40s} median {np.median(v):.4g}")
            else:
                print(f"  {k:40s} median per layer " + " ".join(f"{x:.3g}" for x in np.median(v, 0)))


if __name__ == "__main__":
    main()
