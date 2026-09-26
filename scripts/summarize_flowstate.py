"""Tables for FS1-FS3 (FlowState probes) -> results/flowstate_summary.md."""

import json

import numpy as np


def med(rows, k):
    v = np.array([r[k] for r in rows], dtype=float)
    return np.median(v), v.min(), v.max()


def main():
    L = ["# FlowState probes (released weights, ETTm1/OT, 16 test-period windows, L=1024, H=96)", ""]
    fs = json.load(open("results/flowstate_probe.json"))
    L.append("FS1. Relative change of the median forecast at common physical times, and of the last encoder state, vs native. "
             "Median [min, max] over windows. zohR = every sample repeated r times, lossless for FlowState's own right-hold ZOH kernel; "
             "refine = FOH virtual knots; down = every r-th real sample (lossy).")
    L.append("")
    L.append("| revision | params | grid change | lossless for FlowState's rule? | forecast change | state change |")
    L.append("|---|---|---|---|---|---|")
    for rev, rr in fs["revisions"].items():
        rows = rr["rows"]
        for key, name, lossless in (("zohR2", "ZOH repeat x2", "yes"), ("zohR4", "ZOH repeat x4", "yes"),
                                    ("refine2", "FOH virtual knots x2", "no (FOH path)"), ("refine4", "FOH virtual knots x4", "no (FOH path)"),
                                    ("down2", "real downsampling /2", "no (lossy)"), ("down4", "real downsampling /4", "no (lossy)")):
            f = med(rows, f"{key}_rel_forecast_change")
            s = med(rows, f"{key}_rel_state_change")
            L.append(f"| {rev} | {rr['n_params'] / 1e6:.1f}M | {name} | {lossless} | {100 * f[0]:.1f}% [{100 * f[1]:.1f}, {100 * f[2]:.1f}] | "
                     f"{100 * s[0]:.1f}% [{100 * s[1]:.1f}, {100 * s[2]:.1f}] |")
        rep = max(r["repeat_rel_change"] for r in rows)
        m0, mr = med(rows, "mae_native")[0], med(rows, "refine2_mae")[0]
        L.append(f"| {rev} | | repeat of native input | yes | {rep:.1e} | |")
        L.append("")
        L.append(f"{rev}: forecast MAE native {m0:.3f} vs FOH-refined x2 {mr:.3f} (median over windows).")
        L.append("")
    fa = json.load(open("results/flowstate_attribution.json"))
    L.append("FS2. Attribution on 8 windows: causal RevIN (running sample-count mean/std) vs fixed per-context statistics "
             "(affine, grid-independent). Per-layer relative change of the encoder output at the last context step.")
    L.append("")
    L.append("| revision | RevIN | grid change | forecast change | per-layer change at last step (layer 0 -> 5) |")
    L.append("|---|---|---|---|---|")
    for rev, rows in fa["revisions"].items():
        for mode in ("causal_revin", "fixed_revin"):
            for vn in ("zohR2", "foh2"):
                f = np.median([r[f"{mode}__{vn}__forecast"] for r in rows])
                pl = np.median(np.array([r[f"{mode}__{vn}__per_layer_last_step"] for r in rows]), 0)
                L.append(f"| {rev} | {mode} | {vn} | {100 * f:.1f}% | " + " ".join(f"{100 * x:.2g}%" for x in pl) + " |")
    L.append("")
    f3 = json.load(open("results/flowstate_fp64_layers.json"))
    L.append("FS3. Encoder in float32 vs float64 (weights cast), ZOH repeat x2, fixed RevIN statistics, 4 windows: "
             "median per-layer relative change at the last context step.")
    L.append("")
    L.append("| revision | dtype | layer 0 | 1 | 2 | 3 | 4 | 5 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rev, rows in f3.items():
        for dt in ("torch.float32", "torch.float64"):
            v = np.median(np.array([r["per_layer_rel_change"] for r in rows if r["dtype"] == dt]), 0)
            L.append(f"| {rev} | {dt.split('.')[1]} | " + " | ".join(f"{x:.1e}" for x in v) + " |")
    open("results/flowstate_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
