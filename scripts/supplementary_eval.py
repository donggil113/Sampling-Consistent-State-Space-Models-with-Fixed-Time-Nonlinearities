"""Supplementary eval-only analyses added after the adversarial code review (deviation D7). No retraining.

(1) Rule-matched lossy twins.  For the right-hold (ZOH-rule) models A2_zoh_clock, A2_zoh_clock4, A3_zoh_perobs and
    B_dtonly, the pre-registered twins (FOH-valued virtual knots) are NOT the same path.  Evaluate the ZOH twins
    down2_reknot_zoh and drop50_reknot_zoh (float32, full test set) and the float64 exactness down2_reknot_zoh vs down2.

(2) H2 decomposition for the per-observation rules A1_foh_perobs, A3_zoh_perobs and B_dtonly under the lossless
    refinements foh_m2, foh_m4, foh_rand and zoh_m2.  The per-observation readout is a time-weighted (right Riemann)
    mean over knots, which is itself grid-dependent.  With z_k = head(x_k) the per-knot logits (pooled = sum_k w_k z_k):
      full        : pooled with the refined grid's weights                         (the pre-registered H2 quantity)
      same_times  : pooled only at MEASURED knots (= the native times) with the native weights, from the refined run
                    -> change of the representation at the same physical times
      control     : the NATIVE per-knot logits, interpolated to the refined knots with the model's own rule and
                    pooled with the refined weights -> pure quadrature effect of a grid-free representation
    Relative change = ||pooled - pooled_native||_inf / ||pooled_native||_inf per window (median, p95), flip rate.
    float32 on the full test set; float64 on the first 256 test windows (the H2 float64 median).

Writes results/har_supplementary.json and results/raw/har_supp/<group>__seed<s>__twins_logits.npz.
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
from fxclock.prep import collate  # noqa: E402
from fxclock.train import prep_all  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from run_har import GROUPS  # noqa: E402

torch.set_num_threads(int(os.environ.get("SUPP_THREADS", "1")))
RAW = "results/raw/har"
OUT_DIR = "results/raw/har_supp"
ZOH_RULE = {"A2_zoh_clock", "A2_zoh_clock4", "A3_zoh_perobs", "B_dtonly"}
PER_OBS = {"A1_foh_perobs": "foh", "A3_zoh_perobs": "zoh", "B_dtonly": "zoh"}
LOSSLESS = ["foh_m2", "foh_m4", "foh_rand", "zoh_m2"]
TWINS = ["down2_reknot_zoh", "drop50_reknot_zoh"]


@torch.no_grad()
def per_step(net, items, bs, dtype):
    """Per-knot logits and pooling ingredients for a list of prepared items (per-observation family)."""
    net.eval()
    Z, W, Q, M = [], [], [], []
    for s in range(0, len(items), bs):
        b = collate(items[s:s + bs], dtype=dtype)
        z = net(b, force_scan=False, per_step=True).double().numpy()
        for i in range(z.shape[0]):
            n = items[s + i]["q"].shape[0]
            Z.append(z[i, :n])
            W.append(items[s + i]["w"])
            Q.append(items[s + i]["q"])
            M.append(items[s + i]["measured"])
    return Z, W, Q, M


def measured_weights(q, m, t_start=0.0):
    tm = q[m]
    dt = np.diff(np.concatenate([[t_start], tm]))
    w = np.zeros(q.shape[0])
    w[m] = dt / dt.sum()
    return w


def interp_rule(q0, z0, q, rule):
    if rule == "foh":
        return np.stack([np.interp(q, q0, z0[:, c]) for c in range(z0.shape[1])], axis=1)
    idx = np.clip(np.searchsorted(q0, q, side="left"), 0, q0.shape[0] - 1)   # right hold
    return z0[idx]


def rel(p, p0):
    return np.abs(p - p0).max(1) / np.maximum(np.abs(p0).max(1), 1e-12)


def decompose(net, X, spec, ctx, rule, bs, dtype):
    base_items, _ = prep_all(X, spec, ctx, "native")
    Z0, W0, Q0, _ = per_step(net, base_items, bs, dtype)
    P0 = np.stack([w @ z for w, z in zip(W0, Z0)])
    out = {}
    for cond in LOSSLESS:
        items, _ = prep_all(X, spec, ctx, cond)
        Z, W, Q, M = per_step(net, items, bs, dtype)
        full = np.stack([w @ z for w, z in zip(W, Z)])
        same = np.stack([measured_weights(q, m) @ z for q, m, z in zip(Q, M, Z)])
        ctrl = np.stack([w @ interp_rule(q0, z0, q, rule) for w, q, q0, z0 in zip(W, Q, Q0, Z0)])
        res = {}
        for name, P in (("full", full), ("same_times", same), ("control", ctrl)):
            r = rel(P, P0)
            res[name] = {"median": float(np.median(r)), "p95": float(np.percentile(r, 95)), "max": float(r.max()),
                         "flip_rate": float((P.argmax(1) != P0.argmax(1)).mean())}
        out[cond] = res
    return out


def main():
    cfg_all = json.load(open("configs/prereg_n2.json"))
    cfg = dict(cfg_all["model"])
    cfg.update(cfg_all["training"])
    d = load(cfg_all["data"]["har_root"], cfg_all["data"]["har_cache"])
    mu, sd = d["Xtr"].mean((0, 1)), d["Xtr"].std((0, 1))
    Xte, yte = (d["Xte"] - mu) / sd, d["yte"]
    ctx = {"t_start": 0.0, "t_end": 127 * NATIVE_DT, "native_dt": NATIVE_DT}
    n64 = cfg_all["eval"]["fp64_subset"]
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {"twins": {}, "decomposition": {}}
    for g in ("G_foh1", "G_zoh1", "G_none1", "G_zoh4"):
        for seed in (0, 1, 2):
            ck = os.path.join(RAW, f"{g}__seed{seed}.pt")
            if not os.path.exists(ck):
                print("missing", ck)
                continue
            state = torch.load(ck)
            logits = {}
            for name in GROUPS[g]:
                spec = SPECS[name]
                t0 = time.perf_counter()
                net = Net(spec, 9, 6, cfg, seed=seed, native_dt=NATIVE_DT)
                net.load_state_dict(state)
                net64 = Net(spec, 9, 6, cfg, seed=seed, native_dt=NATIVE_DT).double()
                net64.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in state.items()})
                bs = cfg_all["eval"]["batch_size_perobs"] if spec["family"] == "perobs" else cfg["eval_batch_size"]
                if name in ZOH_RULE:
                    tw = {}
                    for cond in TWINS:
                        items, _ = prep_all(Xte, spec, ctx, cond)
                        from fxclock.train import predict
                        lg = predict(net, items, bs, force_scan=False)
                        logits[f"{name}__{cond}"] = lg.astype(np.float32)
                        tw[cond] = {"acc": float((lg.argmax(1) == yte).mean())}
                    it_t, _ = prep_all(Xte[:n64], spec, ctx, "down2_reknot_zoh")
                    it_p, _ = prep_all(Xte[:n64], spec, ctx, "down2")
                    from fxclock.train import predict
                    a = predict(net64, it_t, bs, dtype=torch.float64, force_scan=False)
                    b = predict(net64, it_p, bs, dtype=torch.float64, force_scan=False)
                    tw["fp64_down2_reknot_zoh_vs_down2"] = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
                    res["twins"].setdefault(name, {})[str(seed)] = tw
                if name in PER_OBS:
                    dec32 = decompose(net, Xte, spec, ctx, PER_OBS[name], bs, torch.float32)
                    dec64 = decompose(net64, Xte[:n64], spec, ctx, PER_OBS[name], bs, torch.float64)
                    res["decomposition"].setdefault(name, {})[str(seed)] = {"float32_full_test": dec32, "float64_subset": dec64}
                print(f"{g} seed {seed} {name}: {time.perf_counter() - t0:.0f}s", flush=True)
                json.dump(res, open("results/har_supplementary.json", "w"), indent=1)
            if logits:
                np.savez_compressed(os.path.join(OUT_DIR, f"{g}__seed{seed}__twins_logits.npz"), **logits)
    json.dump(res, open("results/har_supplementary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
