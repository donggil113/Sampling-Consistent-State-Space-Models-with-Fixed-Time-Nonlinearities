"""Aggregate N2-HAR-MAIN (+ supplementary eval): accuracy with subject-cluster bootstrap, consistency, float64 exactness,
rule-matched artifact share, H2 decomposition, and the pre-registered hypotheses / decision rule evaluated in code.

Outputs results/har_summary.json and results/har_summary.md.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fxclock.data_har import load  # noqa: E402

RAW = "results/raw/har"
SUPP_RAW = "results/raw/har_supp"
CFG = json.load(open("configs/prereg_n2.json"))
LOSSY = ["down2", "down4", "drop30", "drop50", "drop70"]
LOSSLESS_FOH = ["foh_m2", "foh_m4", "foh_rand"]
SIMPLE_RESAMPLING = ["B_point_clock1", "B_point_clock4", "B_binmean_clock4", "B_patch_clock4", "B_tf_clock1", "B_tfpatch_clock4"]
CLOSE_WORK = ["B_nrde_clock4", "B_rformer_clock4", "B_bilin_clock4"]
PROPOSED = ["P1_foh_clock", "P4_foh_clock"]
PATH_FUNCTIONALS = ["P1_foh_clock", "P4_foh_clock", "B_point_clock1", "B_point_clock4", "B_binmean_clock4", "B_patch_clock4",
                    "B_tf_clock1", "B_tfpatch_clock4", "B_nrde_clock4", "B_rformer_clock4"]
ZOH_RULE = {"A2_zoh_clock", "A2_zoh_clock4", "A3_zoh_perobs", "B_dtonly"}
H2_CONDS = {"A1_foh_perobs": ["foh_m2", "foh_m4", "foh_rand"], "A3_zoh_perobs": ["zoh_m2"], "B_dtonly": ["zoh_m2"]}
ORDER = ["P1_foh_clock", "A1_foh_perobs", "A2_zoh_clock", "A3_zoh_perobs", "B_point_clock1", "B_dtonly", "B_tf_clock1",
         "P4_foh_clock", "A2_zoh_clock4", "B_point_clock4", "B_binmean_clock4", "B_patch_clock4", "B_tfpatch_clock4",
         "B_nrde_clock4", "B_rformer_clock4", "B_bilin_clock4"]
N_BOOT = 2000


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RAW, "G_*__seed[0-9].json"))):
        rec = json.load(open(f))
        lg = np.load(f.replace(".json", "__logits.npz"))
        supp = f.replace(RAW, SUPP_RAW).replace(".json", "__twins_logits.npz")
        sl = np.load(supp) if os.path.exists(supp) else None
        for spec, ev in rec["eval"].items():
            logits = {k.split("__", 1)[1]: lg[k].astype(np.float64) for k in lg.files if k.startswith(spec + "__")}
            if sl is not None:
                logits.update({k.split("__", 1)[1]: sl[k].astype(np.float64) for k in sl.files if k.startswith(spec + "__")})
            runs.setdefault(spec, {})[rec["seed"]] = {"rec": rec, "ev": ev, "logits": logits}
    return runs


def boot_subject(v, subj, seed=0):
    rng = np.random.default_rng(seed)
    subs = np.unique(subj)
    idx = [np.flatnonzero(subj == s) for s in subs]
    stats = [v[np.concatenate([idx[p] for p in rng.integers(0, len(subs), len(subs))])].mean() for _ in range(N_BOOT)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def rel_change(a, b):
    return np.abs(a - b).max(1) / np.maximum(np.abs(b).max(1), 1e-12)


def main():
    d = load(CFG["data"]["har_root"], CFG["data"]["har_cache"])
    y, subj = d["yte"], d["ste"]
    runs = load_runs()
    supp = json.load(open("results/har_supplementary.json")) if os.path.exists("results/har_supplementary.json") else {}
    cost = json.load(open("results/cost_har.json")) if os.path.exists("results/cost_har.json") else None
    out = {"n_seeds": {k: sorted(v) for k, v in runs.items()}, "models": {}}
    for spec in [s for s in ORDER if s in runs]:
        seeds = runs[spec]
        S = sorted(seeds)
        m = {}
        for c in seeds[S[0]]["logits"]:
            if not all(c in seeds[s]["logits"] for s in S):
                continue
            corr = np.stack([(seeds[s]["logits"][c].argmax(1) == y).astype(float) for s in S])
            rc = np.stack([rel_change(seeds[s]["logits"][c], seeds[s]["logits"]["native"]) for s in S])
            flips = np.stack([(seeds[s]["logits"][c].argmax(1) != seeds[s]["logits"]["native"].argmax(1)).astype(float) for s in S])
            m[c] = {"acc_mean": float(corr.mean()), "acc_by_seed": corr.mean(1).tolist(), "acc_ci95_subject": list(boot_subject(corr.mean(0), subj)),
                    "rel_logit_change_median": float(np.median(rc)), "rel_logit_change_median_by_seed": np.median(rc, 1).tolist(),
                    "rel_logit_change_p95": float(np.percentile(rc, 95)), "flip_rate": float(flips.mean()), "flip_rate_by_seed": flips.mean(1).tolist()}
        # artifact share with the RULE-MATCHED twin (same path for this model's input rule)
        for lossy in ("down2", "drop50"):
            twin = f"{lossy}_reknot_zoh" if spec in ZOH_RULE else f"{lossy}_reknot"
            if all(twin in seeds[s]["logits"] for s in S):
                num = sum(np.abs(seeds[s]["logits"][lossy] - seeds[s]["logits"][twin]).sum() for s in S)
                den = sum(np.abs(seeds[s]["logits"][lossy] - seeds[s]["logits"]["native"]).sum() for s in S)
                m[f"artifact_share_{lossy}"] = {"twin": twin, "value": float(num / max(den, 1e-300))}
            else:
                m[f"artifact_share_{lossy}"] = {"twin": twin, "value": None}
        fp = {}
        for s in S:
            for k, v in seeds[s]["ev"]["fp64_max_rel_logit_change"].items():
                fp[k] = max(fp.get(k, 0.0), v)
        if spec in ZOH_RULE and spec in supp.get("twins", {}):
            fp["down2_reknot_zoh"] = max(v["fp64_down2_reknot_zoh_vs_down2"] for v in supp["twins"][spec].values())
        m["fp64_max_rel_logit_change_over_seeds"] = fp
        m["lossy_mean_acc"] = float(np.mean([m[c]["acc_mean"] for c in LOSSY]))
        m["n_params"] = seeds[S[0]]["rec"]["n_params"]
        m["epoch_seconds_median_contended"] = float(np.median([h["epoch_seconds"] for s in S for h in seeds[s]["rec"]["history"]]))
        m["best_dev_acc"] = [seeds[s]["rec"]["best_dev_acc"] for s in S]
        out["models"][spec] = m

    # ---- paired differences on COMMON seeds (per-window correctness averaged over seeds and lossy conditions)
    def corr(spec, conds, seeds):
        return np.mean([np.mean([(runs[spec][s]["logits"][c].argmax(1) == y).astype(float) for s in seeds], 0) for c in conds], 0)

    diffs = {}
    for p in PROPOSED:
        for b in SIMPLE_RESAMPLING + CLOSE_WORK + ["A2_zoh_clock", "A2_zoh_clock4", "B_dtonly", "A1_foh_perobs", "A3_zoh_perobs"]:
            if p not in runs or b not in runs:
                continue
            common = sorted(set(runs[p]) & set(runs[b]))
            dl = corr(p, LOSSY, common) - corr(b, LOSSY, common)
            dn = corr(p, ["native"], common) - corr(b, ["native"], common)
            diffs[f"{p} - {b}"] = {"seeds": common, "lossy_mean_diff_pp": 100 * float(dl.mean()), "lossy_ci95_pp": [100 * v for v in boot_subject(dl, subj)],
                                   "native_diff_pp": 100 * float(dn.mean()), "native_ci95_pp": [100 * v for v in boot_subject(dn, subj)]}
    out["paired_differences"] = diffs

    # ---- cost (idle machine, 1 thread)
    costs = {}
    if cost:
        for r in cost["rows"]:
            costs.setdefault(r["model"], {})[r["condition"]] = r
    out["cost"] = costs

    # ---- pre-registered hypotheses
    H = {}
    h1 = {}
    for spec in PATH_FUNCTIONALS:
        if spec not in out["models"]:
            continue
        m = out["models"][spec]
        fp = max(m["fp64_max_rel_logit_change_over_seeds"].get(c, np.inf) for c in LOSSLESS_FOH)
        fl = max(m[c]["flip_rate"] for c in LOSSLESS_FOH)
        h1[spec] = {"fp64_max": fp, "fp32_flip_max": fl, "pass": bool(fp <= 1e-10 and fl <= 1e-3)}
    H["H1_exact"] = {"per_model": h1, "pass": all(v["pass"] for v in h1.values()) and len(h1) == len(PATH_FUNCTIONALS)}
    h2 = {}
    for spec, conds in H2_CONDS.items():
        if spec not in out["models"]:
            continue
        m = out["models"][spec]
        med32 = {c: m[c]["rel_logit_change_median"] for c in conds}
        dec = supp.get("decomposition", {}).get(spec, {})
        med64 = {c: (float(np.median([dec[s]["float64_subset"][c]["full"]["median"] for s in dec])) if dec else None) for c in conds}
        same32 = {c: (float(np.median([dec[s]["float32_full_test"][c]["same_times"]["median"] for s in dec])) if dec else None) for c in conds}
        ctrl32 = {c: (float(np.median([dec[s]["float32_full_test"][c]["control"]["median"] for s in dec])) if dec else None) for c in conds}
        practical = any(m[c]["flip_rate"] >= 5e-3 or abs(m[c]["acc_mean"] - m["native"]["acc_mean"]) >= 5e-3 for c in conds)
        numeric = all(v >= 1e-3 for v in med32.values()) and all(v is not None and v >= 1e-3 for v in med64.values())
        h2[spec] = {"median_fp32": med32, "median_fp64_subset": med64, "same_times_fp32": same32, "readout_control_fp32": ctrl32,
                    "flip_rate": {c: m[c]["flip_rate"] for c in conds}, "numeric_pass": bool(numeric), "practical_pass": bool(practical)}
    H["H2_core"] = {"per_model": h2, "numeric_pass": all(v["numeric_pass"] for v in h2.values()) and len(h2) == 3,
                    "practical_pass": all(v["practical_pass"] for v in h2.values()) and len(h2) == 3}
    if "B_bilin_clock4" in out["models"]:
        H["H3_stem_bilinear"] = {"median_foh_m4": out["models"]["B_bilin_clock4"]["foh_m4"]["rel_logit_change_median"]}
        H["H3_stem_bilinear"]["pass"] = H["H3_stem_bilinear"]["median_foh_m4"] >= 1e-4
    out["hypotheses"] = H

    # ---- decision rule
    G1 = {}
    for p in PROPOSED:
        if p not in out["models"] or not costs.get(p):
            continue
        cp = costs[p]["native"]["total_us_per_window"]
        comps = {}
        for b in SIMPLE_RESAMPLING:
            if b not in out["models"] or not costs.get(b):
                continue
            cb = costs[b]["native"]["total_us_per_window"]
            if cb <= 1.5 * cp:
                dd = diffs[f"{p} - {b}"]
                comps[b] = {"cost_ratio": cb / cp, "lossy_diff_pp": dd["lossy_mean_diff_pp"], "lossy_ci95_pp": dd["lossy_ci95_pp"],
                            "native_diff_pp": dd["native_diff_pp"],
                            "beats": bool(dd["lossy_mean_diff_pp"] >= 1.0 and dd["lossy_ci95_pp"][0] > 0 and dd["native_diff_pp"] >= -0.5)}
        G1[p] = {"own_cost_us": cp, "comparators_within_1.5x_cost": comps,
                 "pass": bool(comps) and all(v["beats"] for v in comps.values())}
    g1 = any(v["pass"] for v in G1.values()) if G1 else None
    close = {}
    for p in PROPOSED:
        for b in CLOSE_WORK:
            k = f"{p} - {b}"
            if k in diffs:
                close[k] = {"lossy_diff_pp": diffs[k]["lossy_mean_diff_pp"], "within_1pp": bool(diffs[k]["lossy_mean_diff_pp"] < 1.0)}
    g2_lit = False   # RELATED_WORK.md: > 20 sources with overall overlap 'substantial' after adversarial re-check
    g2_close = not any(v["within_1pp"] for v in close.values()) if close else None
    out["decision"] = {"G1_beats_simple_resampling": {"per_proposed": G1, "pass": g1},
                       "G2_not_prior_work": {"literature_pass": g2_lit, "close_work_margins": close, "close_work_pass": g2_close,
                                             "pass": bool(g2_lit and g2_close)},
                       "verdict": "GO" if (g1 and g2_lit and g2_close) else "NO_GO"}
    json.dump(out, open("results/har_summary.json", "w"), indent=1)
    write_md(out, supp)


def f1(x):
    return "—" if x is None else f"{100 * x:.1f}"


def write_md(out, supp):
    M = out["models"]
    L = ["# N2-HAR-MAIN summary (generated by scripts/analyze_har.py)", ""]
    L.append("Accuracy (%) on the 2947 test windows, mean over seeds. 95% CI: cluster bootstrap over the 9 test subjects (seed-mean). "
             "Every model was trained on the native grid only.")
    L.append("")
    show = ["native", "foh_m2", "foh_m4", "foh_rand", "zoh_m2", "down2", "down4", "drop30", "drop50", "drop70"]
    L.append("| model | seeds | " + " | ".join(show) + " | lossy mean |")
    L.append("|---" * (len(show) + 3) + "|")
    for spec, m in M.items():
        L.append(f"| {spec} | {len(out['n_seeds'][spec])} | " + " | ".join(f1(m[c]['acc_mean']) if c in m else "—" for c in show)
                 + f" | {100 * m['lossy_mean_acc']:.1f} |")
    L += ["", "Native accuracy with subject CI, and by seed:", ""]
    for spec, m in M.items():
        ci = m["native"]["acc_ci95_subject"]
        L.append(f"- {spec}: {100 * m['native']['acc_mean']:.2f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]; seeds "
                 + ", ".join(f"{100 * a:.2f}" for a in m["native"]["acc_by_seed"]) + f"; params {m['n_params']}")
    L += ["", "Consistency vs native (float32): median relative logit change / flip rate (%). Artifact share uses the twin that is the "
          "same path for the model's own input rule (ZOH twin for A2*, A3, B_dtonly; FOH twin otherwise).", ""]
    show2 = ["foh_m2", "foh_m4", "foh_rand", "zoh_m2", "down2", "drop50"]
    L.append("| model | " + " | ".join(show2) + " | artifact share down2 | artifact share drop50 |")
    L.append("|---" * (len(show2) + 3) + "|")
    for spec, m in M.items():
        a2, a5 = m["artifact_share_down2"]["value"], m["artifact_share_drop50"]["value"]
        L.append(f"| {spec} | " + " | ".join(f"{m[c]['rel_logit_change_median']:.1e} / {100 * m[c]['flip_rate']:.2f}" for c in show2)
                 + f" | {'—' if a2 is None else f'{a2:.3f}'} | {'—' if a5 is None else f'{a5:.3f}'} |")
    L += ["", "float64 max relative logit change over the 256-window subset, max over seeds:", ""]
    for spec, m in M.items():
        L.append(f"- {spec}: " + ", ".join(f"{k} {v:.1e}" for k, v in m["fp64_max_rel_logit_change_over_seeds"].items()))
    dec = supp.get("decomposition", {})
    if dec:
        L += ["", "H2 decomposition (per-observation rules; medians over windows, then median over seeds). full = pre-registered pooled change; "
              "same_times = representation change at the native (measured) times; control = readout quadrature effect of a grid-free "
              "representation (native per-knot logits interpolated with the model's rule, pooled on the refined grid).", ""]
        L.append("| model | condition | full fp32 | same_times fp32 | control fp32 | full fp64 (256) | same_times fp64 | flip full (%) | flip same_times (%) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for spec, bys in dec.items():
            for c in ["foh_m2", "foh_m4", "foh_rand", "zoh_m2"]:
                g = lambda part, key, where: float(np.median([bys[s][where][c][part][key] for s in bys]))
                L.append(f"| {spec} | {c} | {g('full', 'median', 'float32_full_test'):.2e} | {g('same_times', 'median', 'float32_full_test'):.2e} | "
                         f"{g('control', 'median', 'float32_full_test'):.2e} | {g('full', 'median', 'float64_subset'):.2e} | "
                         f"{g('same_times', 'median', 'float64_subset'):.2e} | {100 * g('full', 'flip_rate', 'float32_full_test'):.2f} | "
                         f"{100 * g('same_times', 'flip_rate', 'float32_full_test'):.2f} |")
    L += ["", "Paired differences (pp), proposed minus other, on common seeds; subject-bootstrap 95% CI:", ""]
    for k, v in out["paired_differences"].items():
        L.append(f"- {k}: lossy mean {v['lossy_mean_diff_pp']:+.2f} [{v['lossy_ci95_pp'][0]:+.2f}, {v['lossy_ci95_pp'][1]:+.2f}]; "
                 f"native {v['native_diff_pp']:+.2f} [{v['native_ci95_pp'][0]:+.2f}, {v['native_ci95_pp'][1]:+.2f}]")
    if out["cost"]:
        L += ["", "Cost including preprocessing (idle machine, 1 thread, 512 test windows; microseconds per window):", ""]
        L.append("| model | params | cond | mean length | prep | forward | total |")
        L.append("|---|---|---|---|---|---|---|")
        for spec in M:
            for c, r in out["cost"].get(spec, {}).items():
                L.append(f"| {spec} | {r['n_params']} | {c} | {r['mean_len']:.0f} | {r['prep_us_per_window']:.0f} | {r['fwd_us_per_window']:.0f} | {r['total_us_per_window']:.0f} |")
    L += ["", "## Hypotheses and decision (evaluated in code)", "", "```", json.dumps({"hypotheses": out["hypotheses"], "decision": out["decision"]}, indent=1), "```"]
    open("results/har_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
