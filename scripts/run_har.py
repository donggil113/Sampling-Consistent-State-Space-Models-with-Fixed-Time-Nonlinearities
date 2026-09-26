"""Train one training group on UCI-HAR (native grid only) and evaluate its inference rules on all test conditions.

Usage: python3 scripts/run_har.py --group G_foh1 --seed 0 --config configs/prereg_n2.json --out results/raw/har
       python3 scripts/run_har.py --group G_foh1 --seed 99 --pilot ...   (dev only; test untouched)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fxclock.conditions import CONDITIONS  # noqa: E402
from fxclock.data_har import NATIVE_DT, dev_subjects, load  # noqa: E402
from fxclock.models import SPECS, Net  # noqa: E402
from fxclock.train import predict, prep_all, train  # noqa: E402

# One training per group; every listed inference rule is evaluated with the SAME trained weights.
# On the native grid the clocked and per-observation rules of a group define the same function
# (clock = native knot times), so training them separately would only add numerical noise.
GROUPS = {
    "G_foh1": ["P1_foh_clock", "A1_foh_perobs"],
    "G_zoh1": ["A2_zoh_clock", "A3_zoh_perobs"],
    "G_none1": ["B_point_clock1", "B_dtonly"],
    "G_tf1": ["B_tf_clock1"],
    "G_P4": ["P4_foh_clock"],
    "G_zoh4": ["A2_zoh_clock4"],
    "G_pt4": ["B_point_clock4"],
    "G_bin4": ["B_binmean_clock4"],
    "G_patch4": ["B_patch_clock4"],
    "G_tfpatch4": ["B_tfpatch_clock4"],
    "G_nrde4": ["B_nrde_clock4"],
    "G_rformer4": ["B_rformer_clock4"],
    "G_bilin4": ["B_bilin_clock4"],
}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--eval-only", action="store_true", help="load the saved checkpoint and training record; skip training")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg_all = json.load(open(a.config))
    cfg = dict(cfg_all["model"])
    cfg.update(cfg_all["training"])
    specs = [SPECS[n] for n in GROUPS[a.group]]
    train_spec = specs[0]
    fam = "transformer" if train_spec["backbone"] == "transformer" else ("nrde" if train_spec["family"] == "nrde" else "ssm")
    cfg["lr"] = a.lr if a.lr is not None else cfg["lr_by_family"][fam]
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    tag = f"{a.group}__seed{a.seed}" + ("__pilot" if a.pilot else "") + (f"__lr{cfg['lr']:g}" if a.lr is not None else "")
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, tag + (".eval.log" if a.eval_only else ".log")), "w")

    def log(s):
        print(s, file=logf, flush=True)

    d = load(cfg_all["data"]["har_root"], cfg_all["data"]["har_cache"])
    mu, sd = d["Xtr"].mean((0, 1)), d["Xtr"].std((0, 1))
    dev_s = dev_subjects(d["str"], cfg_all["data"]["n_dev_subjects"], cfg_all["data"]["dev_seed"])
    is_dev = np.isin(d["str"], dev_s)
    Xtr, ytr = (d["Xtr"][~is_dev] - mu) / sd, d["ytr"][~is_dev]
    Xdv, ydv = (d["Xtr"][is_dev] - mu) / sd, d["ytr"][is_dev]
    Xte, yte = (d["Xte"] - mu) / sd, d["yte"]
    L = d["Xtr"].shape[1]
    ctx = {"t_start": 0.0, "t_end": (L - 1) * NATIVE_DT, "native_dt": NATIVE_DT}
    log(f"group {a.group} seed {a.seed} lr {cfg['lr']} epochs {cfg['epochs']} dev subjects {dev_s.tolist()} "
        f"n_train {len(ytr)} n_dev {len(ydv)} n_test {len(yte)}")

    tr_items, tp_tr = prep_all(Xtr, train_spec, ctx)
    dv_items, _ = prep_all(Xdv, train_spec, ctx)
    ls_scale = None
    if train_spec["family"] == "nrde" or any(f.startswith("logsig") for f in train_spec["feats"]):
        ls_all = np.concatenate([it["ls" if "ls" in it else "feats"] for it in tr_items])
        ls_scale = np.sqrt((ls_all ** 2).mean(0)) + 1e-12   # RMS (a std is 0 for the constant time increment)
    net = Net(train_spec, Xtr.shape[2], 6, cfg, seed=a.seed, native_dt=NATIVE_DT, ls_scale=ls_scale)
    n_params = sum(p.numel() for p in net.parameters())
    if a.eval_only:
        net.load_state_dict(torch.load(os.path.join(a.out, tag + ".pt")))
        trp = os.path.join(a.out, tag + "__train.json")
        if os.path.exists(trp):
            rec = json.load(open(trp))
        else:  # runs trained before the __train.json record existed: parse the training log
            hist = []
            for line in open(os.path.join(a.out, tag + ".log")):
                p = line.split()
                if len(p) >= 7 and p[0] == "ep":
                    hist.append({"epoch": int(p[1]), "train_loss": float(p[3]), "dev_acc": float(p[5]),
                                 "epoch_seconds": float(p[6].strip("(s)"))})
            rec = {"group": a.group, "seed": a.seed, "pilot": False, "cfg": cfg, "n_params": n_params,
                   "best_dev_acc": max(h["dev_acc"] for h in hist), "history": hist,
                   "train_seconds": sum(h["epoch_seconds"] for h in hist), "prep_train_seconds": tp_tr, "threads": a.threads,
                   "git_head_train": "see log (trained before eval-only split)", "config_sha256": sha(a.config)}
        rec["eval_only_git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    else:
        t0 = time.perf_counter()
        hist, best_dev = train(net, tr_items, ytr, dv_items, ydv, cfg, a.seed, log=log)
        t_train = time.perf_counter() - t0
    rec = rec if a.eval_only else {"group": a.group, "seed": a.seed, "pilot": a.pilot, "cfg": cfg, "n_params": n_params, "best_dev_acc": best_dev,
           "history": hist, "train_seconds": t_train, "prep_train_seconds": tp_tr, "threads": a.threads,
           "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "config_sha256": sha(a.config)}
    if a.pilot:
        json.dump(rec, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
        return
    if not a.eval_only:
        torch.save(net.state_dict(), os.path.join(a.out, tag + ".pt"))
        json.dump(rec, open(os.path.join(a.out, tag + "__train.json"), "w"), indent=1)
    sub_idx = np.arange(cfg_all["eval"]["fp64_subset"])
    rec["eval"] = {}
    logits_out = {}
    for spec in specs:
        enet = Net(spec, Xtr.shape[2], 6, cfg, seed=a.seed, native_dt=NATIVE_DT, ls_scale=ls_scale)
        enet.load_state_dict(net.state_dict())
        bs = cfg_all["eval"]["batch_size_perobs"] if spec["family"] == "perobs" else cfg["eval_batch_size"]
        ev = {}
        for cond in CONDITIONS:
            items, tp = prep_all(Xte, spec, ctx, cond)
            t1 = time.perf_counter()
            lg = predict(enet, items, bs)
            ev[cond] = {"prep_seconds": tp, "forward_seconds": time.perf_counter() - t1,
                        "acc": float((lg.argmax(1) == yte).mean()),
                        "mean_knots": float(np.mean([it["q"].shape[0] for it in items]))}
            logits_out[f"{spec['name']}__{cond}"] = lg.astype(np.float32)
            log(f"  eval {spec['name']:18s} {cond:14s} acc {ev[cond]['acc']:.4f}")
        # float64 exactness check on a fixed subset (lossless conditions)
        enet64 = Net(spec, Xtr.shape[2], 6, cfg, seed=a.seed, native_dt=NATIVE_DT, ls_scale=ls_scale).double()
        enet64.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in net.state_dict().items()})
        base_items, _ = prep_all(Xte[sub_idx], spec, ctx, "native")
        base = predict(enet64, base_items, bs, dtype=torch.float64)
        fp64 = {}
        for cond in ("foh_m2", "foh_m4", "foh_rand", "zoh_m2", "down2_reknot"):
            it, _ = prep_all(Xte[sub_idx], spec, ctx, cond)
            lg = predict(enet64, it, bs, dtype=torch.float64)
            if cond == "down2_reknot":
                it2, _ = prep_all(Xte[sub_idx], spec, ctx, "down2")
                ref = predict(enet64, it2, bs, dtype=torch.float64)
            else:
                ref = base
            fp64[cond] = float(np.max(np.abs(lg - ref)) / np.max(np.abs(ref)))
        ev["fp64_max_rel_logit_change"] = fp64
        log(f"  fp64 {spec['name']}: {fp64}")
        rec["eval"][spec["name"]] = ev
    np.savez_compressed(os.path.join(a.out, tag + "__logits.npz"), **logits_out)
    json.dump(rec, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
