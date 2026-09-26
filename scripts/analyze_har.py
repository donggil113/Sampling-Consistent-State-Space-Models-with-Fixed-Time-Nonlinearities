"""Aggregate N2-HAR-MAIN: accuracy (subject-cluster bootstrap), consistency, fp64 exactness, artifact share, decision rule."""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fxclock.data_har import load  # noqa: E402

RAW = "results/raw/har"
CFG = json.load(open("configs/prereg_n2.json"))
LOSSY = ["down2", "down4", "drop30", "drop50", "drop70"]
LOSSLESS_FOH = ["foh_m2", "foh_m4", "foh_rand"]
SIMPLE_RESAMPLING = ["B_point_clock1", "B_point_clock4", "B_binmean_clock4", "B_patch_clock4", "B_tf_clock1", "B_tfpatch_clock4"]
CLOSE_WORK = ["B_nrde_clock4", "B_rformer_clock4", "B_bilin_clock4"]
PROPOSED = ["P1_foh_clock", "P4_foh_clock"]
PATH_FUNCTIONALS = ["P1_foh_clock", "P4_foh_clock", "B_point_clock1", "B_point_clock4", "B_binmean_clock4", "B_patch_clock4",
                    "B_tf_clock1", "B_tfpatch_clock4", "B_nrde_clock4", "B_rformer_clock4"]
N_BOOT = 2000


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RAW, "G_*__seed[0-9].json"))):
        rec = json.load(open(f))
        lg = np.load(f.replace(".json", "__logits.npz"))
        for spec, ev in rec["eval"].items():
            runs.setdefault(spec, {})[rec["seed"]] = {"rec": rec, "ev": ev,
                                                      "logits": {k.split("__", 1)[1]: lg[k].astype(np.float64) for k in lg.files if k.startswith(spec + "__")}}
    return runs


def boot_subject(values_by_window, subj, rng_seed=0):
    """values_by_window: (n,) per-window statistic (already seed-averaged). Cluster bootstrap over subjects of the mean."""
    rng = np.random.default_rng(rng_seed)
    subs = np.unique(subj)
    idx = [np.flatnonzero(subj == s) for s in subs]
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(subs), len(subs))
        w = np.concatenate([idx[p] for p in pick])
        stats.append(values_by_window[w].mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def rel_change(a, b):
    return np.abs(a - b).max(1) / np.maximum(np.abs(b).max(1), 1e-12)


def main():
    d = load(CFG["data"]["har_root"], CFG["data"]["har_cache"])
    y, subj = d["yte"], d["ste"]
    runs = load_runs()
    conds = list(next(iter(next(iter(runs.values())).values()))["logits"].keys())
    out = {"n_seeds": {k: len(v) for k, v in runs.items()}, "models": {}}
    for spec, seeds in runs.items():
        m = {}
        for c in conds:
            corr = np.stack([(seeds[s]["logits"][c].argmax(1) == y).astype(float) for s in sorted(seeds)])   # (S, n)
            acc_seed = corr.mean(1)
            lo, hi = boot_subject(corr.mean(0), subj)
            rc = np.stack([rel_change(seeds[s]["logits"][c], seeds[s]["logits"]["native"]) for s in sorted(seeds)])
            flips = np.stack([(seeds[s]["logits"][c].argmax(1) != seeds[s]["logits"]["native"].argmax(1)).astype(float)
                              for s in sorted(seeds)])
            m[c] = {"acc_mean": float(acc_seed.mean()), "acc_by_seed": acc_seed.tolist(), "acc_ci95_subject": [lo, hi],
                    "rel_logit_change_median": float(np.median(rc)), "rel_logit_change_p95": float(np.percentile(rc, 95)),
                    "flip_rate": float(flips.mean()), "flip_rate_by_seed": flips.mean(1).tolist()}
        # artifact share (same path, different knots) for lossy conditions with a re-knotted twin
        for lossy, twin in (("down2", "down2_reknot"), ("drop50", "drop50_reknot")):
            num = np.concatenate([np.abs(seeds[s]["logits"][lossy] - seeds[s]["logits"][twin]).sum(1) for s in sorted(seeds)])
            den = np.concatenate([np.abs(seeds[s]["logits"][lossy] - seeds[s]["logits"]["native"]).sum(1) for s in sorted(seeds)])
            m[f"artifact_share_{lossy}"] = float(num.sum() / max(den.sum(), 1e-300))
        m["fp64_max_rel_logit_change"] = {s: seeds[s]["ev"]["fp64_max_rel_logit_change"] for s in sorted(seeds)}
        m["lossy_mean_acc"] = float(np.mean([m[c]["acc_mean"] for c in LOSSY]))
        m["n_params"] = seeds[sorted(seeds)[0]]["rec"]["n_params"]
        m["train_seconds_mean"] = float(np.mean([seeds[s]["rec"]["train_seconds"] for s in seeds]))
        m["epoch_seconds_median"] = float(np.median([h["epoch_seconds"] for s in seeds for h in seeds[s]["rec"]["history"]]))
        m["best_dev_acc"] = [seeds[s]["rec"]["best_dev_acc"] for s in sorted(seeds)]
        out["models"][spec] = m

    # paired differences P - baseline on the lossy mean (per window, seed-averaged), subject bootstrap
    def lossy_corr(spec):
        seeds = runs[spec]
        return np.mean([np.mean([(seeds[s]["logits"][c].argmax(1) == y).astype(float) for s in sorted(seeds)], 0) for c in LOSSY], 0)

    def native_corr(spec):
        seeds = runs[spec]
        return np.mean([(seeds[s]["logits"]["native"].argmax(1) == y).astype(float) for s in sorted(seeds)], 0)

    diffs = {}
    for p in PROPOSED:
        if p not in runs:
            continue
        for b in SIMPLE_RESAMPLING + CLOSE_WORK:
            if b not in runs:
                continue
            dl = lossy_corr(p) - lossy_corr(b)
            dn = native_corr(p) - native_corr(b)
            diffs[f"{p} - {b}"] = {"lossy_mean_diff_pp": 100 * float(dl.mean()), "lossy_ci95_pp": [100 * v for v in boot_subject(dl, subj)],
                                   "native_diff_pp": 100 * float(dn.mean()), "native_ci95_pp": [100 * v for v in boot_subject(dn, subj)]}
    out["paired_differences"] = diffs
    cost_path = "results/cost_har.json"
    out["cost"] = json.load(open(cost_path)) if os.path.exists(cost_path) else None
    json.dump(out, open("results/har_summary.json", "w"), indent=1)
    write_md(out, conds)


def fmt_ci(ci):
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def write_md(out, conds):
    L = ["# N2-HAR-MAIN summary (generated by scripts/analyze_har.py)", ""]
    L.append("Accuracy (%), mean over seeds; 95% CI = cluster bootstrap over the 9 test subjects of the seed-mean.")
    L.append("")
    show = ["native", "foh_m2", "foh_m4", "foh_rand", "zoh_m2", "down2", "down4", "drop30", "drop50", "drop70"]
    L.append("| model | seeds | " + " | ".join(show) + " | lossy mean |")
    L.append("|---" * (len(show) + 3) + "|")
    for spec, m in out["models"].items():
        L.append(f"| {spec} | {out['n_seeds'][spec]} | " + " | ".join(f"{100 * m[c]['acc_mean']:.1f}" for c in show)
                 + f" | {100 * m['lossy_mean_acc']:.1f} |")
    L += ["", "Native-grid accuracy with CI:", ""]
    for spec, m in out["models"].items():
        L.append(f"- {spec}: {100 * m['native']['acc_mean']:.2f} {fmt_ci(m['native']['acc_ci95_subject'])}; by seed "
                 + ", ".join(f"{100 * a:.2f}" for a in m["native"]["acc_by_seed"]))
    L += ["", "Consistency vs native: median relative logit change / flip rate (%)", ""]
    show2 = ["foh_m2", "foh_m4", "foh_rand", "zoh_m2", "down2", "down2_reknot", "drop50", "drop50_reknot"]
    L.append("| model | " + " | ".join(show2) + " | artifact share down2 | artifact share drop50 |")
    L.append("|---" * (len(show2) + 3) + "|")
    for spec, m in out["models"].items():
        L.append(f"| {spec} | " + " | ".join(f"{m[c]['rel_logit_change_median']:.1e} / {100 * m[c]['flip_rate']:.2f}" for c in show2)
                 + f" | {m['artifact_share_down2']:.3f} | {m['artifact_share_drop50']:.3f} |")
    L += ["", "float64 max relative logit change on 256 test windows (per seed):", ""]
    for spec, m in out["models"].items():
        vals = {k: max(v[k] for v in m["fp64_max_rel_logit_change"].values()) for k in next(iter(m["fp64_max_rel_logit_change"].values()))}
        L.append(f"- {spec}: " + ", ".join(f"{k} {v:.1e}" for k, v in vals.items()))
    L += ["", "Paired differences (percentage points), proposed minus baseline, subject-bootstrap 95% CI:", ""]
    for k, v in out["paired_differences"].items():
        L.append(f"- {k}: lossy mean {v['lossy_mean_diff_pp']:+.2f} [{v['lossy_ci95_pp'][0]:+.2f}, {v['lossy_ci95_pp'][1]:+.2f}]; "
                 f"native {v['native_diff_pp']:+.2f} [{v['native_ci95_pp'][0]:+.2f}, {v['native_ci95_pp'][1]:+.2f}]")
    open("results/har_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
