"""Training / evaluation loop shared by all models (one code path, one hyper-parameter set)."""

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import knots as K
from .conditions import apply
from .prep import collate, prep_window


def prep_all(Xn, spec, ctx, cond="native", idx_offset=0):
    """Preprocess normalized windows Xn (n, L, C) under a grid condition; returns (items, seconds)."""
    t0 = time.perf_counter()
    items = []
    for i in range(Xn.shape[0]):
        ks = K.from_uniform(Xn[i], ctx["native_dt"])
        items.append(prep_window(apply(cond, ks, idx_offset + i), spec, ctx))
    return items, time.perf_counter() - t0


def batches(items, y, bs, rng=None, dtype=torch.float32):
    order = np.arange(len(items)) if rng is None else rng.permutation(len(items))
    for s in range(0, len(items), bs):
        idx = order[s:s + bs]
        yield collate([items[i] for i in idx], dtype=dtype), torch.as_tensor(y[idx])


@torch.no_grad()
def predict(net, items, bs, dtype=torch.float32, force_scan=True):
    net.eval()
    outs = []
    for b, _ in batches(items, np.zeros(len(items), dtype=np.int64), bs, dtype=dtype):
        outs.append(net(b, force_scan=force_scan).to(torch.float64))
    return torch.cat(outs).numpy()


def train(net, tr_items, ytr, dev_items, ydev, cfg, seed, log=print, items_for_epoch=None):
    """items_for_epoch(ep) -> list of items: optional per-epoch training grids (augmentation arm)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * math.ceil(len(tr_items) / cfg["batch_size"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    best, best_acc, hist = None, -1.0, []
    for ep in range(cfg["epochs"]):
        net.train()
        t0 = time.perf_counter()
        tot, nb = 0.0, 0
        if items_for_epoch is not None:
            tr_items = items_for_epoch(ep)
        for b, yb in batches(tr_items, ytr, cfg["batch_size"], rng):
            loss = F.cross_entropy(net(b), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        t_ep = time.perf_counter() - t0
        dev_logits = predict(net, dev_items, cfg["eval_batch_size"], force_scan=False)
        acc = float((dev_logits.argmax(1) == ydev).mean())
        hist.append({"epoch": ep, "train_loss": tot / nb, "dev_acc": acc, "epoch_seconds": t_ep})
        log(f"  ep {ep:3d} loss {tot / nb:.4f} dev {acc:.4f} ({t_ep:.1f}s)")
        if acc > best_acc:
            best_acc, best = acc, copy.deepcopy(net.state_dict())
    net.load_state_dict(best)
    return hist, best_acc
