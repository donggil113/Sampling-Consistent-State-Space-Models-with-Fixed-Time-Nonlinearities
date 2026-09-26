"""Numerical log for the exact stem (N2-NUM): phi accuracy, refinement floors, small dt, fwd/bwd.

Writes results/numerics.json.  Inputs are real HAR test windows (normalized with train statistics).
"""

import json
import math
import os
import sys
import time

import mpmath
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fxclock import knots as K  # noqa: E402
from fxclock.data_har import NATIVE_DT, load  # noqa: E402
from fxclock.phi import phi12, phi12_naive  # noqa: E402
from fxclock.stem import ExactStem  # noqa: E402

torch.set_num_threads(1)
OUT = "results/numerics.json"


def mp_phi(z):
    mpmath.mp.dps = 60
    zm = mpmath.mpc(z.real, z.imag)
    e = mpmath.exp(zm)
    return complex((e - 1) / zm), complex((e - 1 - zm) / zm ** 2)


def phi_table():
    rows = []
    for r in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 0.49, 0.51, 1.0, 10.0, 100.0):
        z = r * complex(-0.6, 0.8)
        r1, r2 = mp_phi(z)
        for dt, cdt in (("float64", torch.complex128), ("float32", torch.complex64)):
            zt = torch.tensor([z], dtype=cdt)
            for name, fn in (("stable", phi12), ("naive", phi12_naive)):
                p1, p2 = fn(zt)
                rows.append({"abs_z": r, "dtype": dt, "formula": name,
                             "rel_err_phi1": abs(complex(p1[0]) - r1) / abs(r1),
                             "rel_err_phi2": abs(complex(p2[0]) - r2) / abs(r2)})
    return rows


def stem_tensors(ks, q, dtype):
    seg = K.segments(ks, q, "foh", 0.0)
    T = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt)[None]
    return (T(seg["d"]), T(seg["xs"]), T(seg["xe"]), T(seg["win"], torch.long), T(seg["rem"]), T(seg["qdt"])), seg


def sequential_composition(stem, seg, dtype):
    """Reference formulation: h <- e^{lam d} h + c_seg for every segment, in order (N1-style composition)."""
    lam = stem.lam().detach().to(torch.complex128 if dtype == torch.float64 else torch.complex64)
    B = torch.complex(stem.B_re, stem.B_im).detach().to(lam.dtype)
    d = torch.as_tensor(seg["d"], dtype=dtype)
    xs = torch.as_tensor(seg["xs"], dtype=dtype).to(lam.dtype) @ B.T
    xe = torch.as_tensor(seg["xe"], dtype=dtype).to(lam.dtype) @ B.T
    z = d[:, None].to(lam.dtype) * lam
    p1, p2 = phi12(z)
    a = torch.exp(z).numpy()
    c = (d[:, None].to(lam.dtype) * ((p1 - p2) * xs + p2 * xe)).numpy()
    win = seg["win"]
    h = np.zeros(lam.shape[0], dtype=a.dtype)
    out = []
    for k in range(a.shape[0]):
        h = a[k] * h + c[k]
        if k == a.shape[0] - 1 or win[k + 1] != win[k]:
            out.append(h.copy())
    return torch.as_tensor(np.stack(out))


def refinement_floors(X, n_windows=8):
    stem64 = ExactStem(9, 32, 0.02, 2.56, 25.0, seed=0).double()
    stem32 = ExactStem(9, 32, 0.02, 2.56, 25.0, seed=0)
    q = K.clock_times(31, 4, NATIVE_DT)
    rows = []
    for w in range(n_windows):
        ks = K.from_uniform(X[w], NATIVE_DT)
        (a64, seg64) = stem_tensors(ks, q, torch.float64)
        with torch.no_grad():
            h_ref = stem64(*a64)[0]
        for m in (1, 2, 4, 16, 64, 256, 1024, 4096):
            ksm = K.refine_uniform(ks, m, "foh")
            a64m, segm = stem_tensors(ksm, q, torch.float64)
            a32m, _ = stem_tensors(ksm, q, torch.float32)
            with torch.no_grad():
                h64 = stem64(*a64m)[0]
                h32 = stem32(*a32m)[0].to(torch.complex128)
                # naive phi in the window-sum formulation (float32)
                import fxclock.stem as S
                saved = S.phi12
                S.phi12 = phi12_naive
                h32n = stem32(*a32m)[0].to(torch.complex128)
                S.phi12 = saved
            rec = {"window": w, "m": m, "n_segments": int(segm["d"].shape[0]),
                   "min_abs_z": float(np.min(segm["d"][segm["d"] > 0]) * float(stem64.lam().abs().min())),
                   "fp64_window_sum": float((h64 - h_ref).abs().max() / h_ref.abs().max()),
                   "fp32_window_sum": float((h32 - h_ref).abs().max() / h_ref.abs().max()),
                   "fp32_window_sum_naive_phi": float((h32n - h_ref).abs().max() / h_ref.abs().max())}
            if m <= 1024:
                hs = sequential_composition(stem32, segm, torch.float32).to(torch.complex128)
                rec["fp32_sequential"] = float((hs - h_ref).abs().max() / h_ref.abs().max())
            rows.append(rec)
    return rows


def main():
    d = load("/home/user/data/UCI HAR Dataset", "/home/user/data/har_cache.npz")
    mu, sd = d["Xtr"].mean((0, 1)), d["Xtr"].std((0, 1))
    X = (d["Xte"] - mu) / sd
    t0 = time.perf_counter()
    res = {"phi": phi_table(), "refinement": refinement_floors(X)}
    res["seconds"] = time.perf_counter() - t0
    json.dump(res, open(OUT, "w"), indent=1)
    print("phi (worst rel err over phi1/phi2):")
    for r in res["phi"]:
        print(f"  |z|={r['abs_z']:<8g} {r['dtype']} {r['formula']:6s} phi1 {r['rel_err_phi1']:.1e} phi2 {r['rel_err_phi2']:.1e}")
    print("refinement floors (median over windows):")
    for m in (1, 2, 4, 16, 64, 256, 1024, 4096):
        rs = [r for r in res["refinement"] if r["m"] == m]
        keys = ["fp64_window_sum", "fp32_window_sum", "fp32_window_sum_naive_phi", "fp32_sequential"]
        vals = {k: np.median([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan") for k in keys}
        print(f"  m={m:5d} min|z|={np.median([r['min_abs_z'] for r in rs]):.1e} " + " ".join(f"{k}={v:.1e}" for k, v in vals.items()))


if __name__ == "__main__":
    main()
