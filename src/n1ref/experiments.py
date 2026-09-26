"""FIRST_RUN experiments (see configs/first_run.json for the pre-registration)."""

import math
import random
import struct
from decimal import Decimal, getcontext

from .grids import base_grid, base_times, first_half_mask, refine_grid, sample_path, split_grid
from .metrics import (
    at_index,
    diff_l2,
    flat_states,
    l2,
    readouts,
    rel_l2,
    rms,
)
from .model import VARIANTS, ToyParams, run_scan, sample_params
from .reference import ct_held, ct_true


# ----------------------------------------------------------------------------- units

def make_unit(seed, cfg, base_kind="uniform", complex_modes=False):
    """One independent evaluation unit: parameters, input path and base grid."""
    rng = random.Random(seed)
    params = sample_params(rng, n=cfg["model"]["state_dim_N"], complex_modes=complex_modes)
    path = sample_path(rng)
    T, K = cfg["grids"]["T"], cfg["grids"]["K_base"]
    times = base_times(T, K, base_kind, rng=random.Random(10_000 + seed))
    return params, path, base_grid(times, path)


def _tau(base):
    return base.T / base.n_steps


def _config_tag(base_kind, complex_modes):
    return f"{base_kind}-{'complex' if complex_modes else 'real'}"


# ----------------------------------------------------------------------------- E1

def run_e1(seeds, cfg, base_kind="uniform", complex_modes=False):
    """Interval splitting with the held value repeated (physical path unchanged)."""
    tag = _config_tag(base_kind, complex_modes)
    ms = cfg["grids"]["m_values"]
    recs = []
    for seed in seeds:
        params, path, base = make_unit(seed, cfg, base_kind, complex_modes)
        tau = _tau(base)
        exact = run_scan(params, base.times, base.values, "zoh_dt", tau)
        ref = ct_held(params, base)
        y_exact, h_exact = exact["y"], flat_states(exact["h"])
        recs.append({
            "exp": "FR-E1-SPLIT", "config": tag, "seed": seed, "kind": "reference_check",
            "exact_zoh_vs_rk4_rel_l2_y": rel_l2(y_exact, ref["y"]),
            "exact_zoh_vs_rk4_rel_l2_h": rel_l2(h_exact, flat_states(ref["h"])),
            "rk4_total_substeps": ref["total_substeps"],
        })
        for v in VARIANTS:
            y1 = h1 = None
            for m in ms:
                G = split_grid(base, m)
                out = run_scan(params, G.times, G.values, v, tau)
                yb = at_index(out["y"], G.base_index)
                hb = flat_states(at_index(out["h"], G.base_index))
                if m == 1:
                    y1, h1 = yb, hb
                recs.append({
                    "exp": "FR-E1-SPLIT", "config": tag, "seed": seed, "kind": "variant",
                    "variant": v, "m": m, "n_steps": G.n_steps,
                    "change_y": rel_l2(yb, y1), "change_h": rel_l2(hb, h1),
                    "artifact_y": rel_l2(yb, y_exact), "artifact_h": rel_l2(hb, h_exact),
                    "y_base": yb,
                })
    return recs


# ----------------------------------------------------------------------------- E2

def run_e2(seeds, cfg, base_kind="uniform", complex_modes=False):
    """Nested refinement with NEW observations; decompose the output change."""
    tag = _config_tag(base_kind, complex_modes)
    ms = cfg["grids"]["m_values"]
    recs = []
    for seed in seeds:
        params, path, base = make_unit(seed, cfg, base_kind, complex_modes)
        tau = _tau(base)
        truth = ct_true(params, path, base.times, n_sub=512)
        y_true = truth["y"]
        ny = max(l2(y_true), 1e-300)
        grids, cth = {}, {}
        for m in ms:
            grids[m] = refine_grid(base, m, path)
            ref = ct_held(params, grids[m])
            cth[m] = at_index(ref["y"], grids[m].base_index)
            recs.append({
                "exp": "FR-E2-NEWOBS", "config": tag, "seed": seed, "kind": "reference",
                "m": m, "ct_held_err_to_truth": rel_l2(cth[m], y_true),
                "ct_held_path_change": rel_l2(cth[m], cth[1]),
                "rk4_total_substeps": ref["total_substeps"],
            })
        for v in VARIANTS:
            d = {}
            for m in ms:
                G = grids[m]
                out = run_scan(params, G.times, G.values, v, tau)
                d[m] = at_index(out["y"], G.base_index)
            art1 = [a - b for a, b in zip(d[1], cth[1])]
            for m in ms:
                artm = [a - b for a, b in zip(d[m], cth[m])]
                recs.append({
                    "exp": "FR-E2-NEWOBS", "config": tag, "seed": seed, "kind": "variant",
                    "variant": v, "m": m, "n_steps": grids[m].n_steps,
                    "artifact": rel_l2(d[m], cth[m]),
                    "total_change": rel_l2(d[m], d[1]),
                    "path_change": rel_l2(cth[m], cth[1]),
                    "err_to_truth": rel_l2(d[m], y_true),
                    "nd_total": diff_l2(d[m], d[1]) / ny,
                    "nd_path": diff_l2(cth[m], cth[1]) / ny,
                    "nd_artifact_delta": diff_l2(artm, art1) / ny,
                    "y_base": d[m],
                })
    return recs


# ----------------------------------------------------------------------------- E3

def run_e3(seeds, cfg, base_kind="uniform", complex_modes=False):
    """Sample-count vs time-weighted readouts under density changes."""
    tag = _config_tag(base_kind, complex_modes)
    ms = cfg["grids"]["m_values"]
    recs = []
    for seed in seeds:
        params, path, base = make_unit(seed, cfg, base_kind, complex_modes)
        tau = _tau(base)
        mask = first_half_mask(base)
        K1 = base.n_steps

        # (a) nonuniform split: held path unchanged, so the hold-exact time average is fixed
        held_avg = ct_held(params, base)["integral"] / base.T
        for v in VARIANTS:
            r1 = scale = None
            for m in ms:
                G = split_grid(base, m, mask)
                out = run_scan(params, G.times, G.values, v, tau)
                r = readouts(G.times, out["y"], out["integral"])
                if m == 1:
                    r1 = r
                    scale = max(rms(at_index(out["y"], G.base_index)), 1e-300)
                rec = {"exp": "FR-E3-READOUT", "config": tag, "seed": seed, "scenario": "a_split_nonuniform",
                       "variant": v, "m": m, "n_steps": G.n_steps, "scale": scale, "readouts": r}
                for name, val in r.items():
                    if val is None:
                        rec[f"change_{name}"] = None
                        continue
                    s = scale * (K1 if name == "sample_sum" else 1)
                    rec[f"change_{name}"] = abs(val - r1[name]) / s
                    if name in ("sample_mean", "time_riemann", "time_trapz", "time_exact"):
                        rec[f"err_heldavg_{name}"] = abs(val - held_avg) / scale
                if r["time_exact"] is not None:
                    rec["riemann_minus_exact"] = abs(r["time_riemann"] - r["time_exact"]) / scale
                recs.append(rec)

        # (b) nonuniform NEW observations: compare with the continuous-time truth
        truth = ct_true(params, path, base.times, n_sub=512)
        true_avg = truth["integral"] / base.T
        true_last = truth["y"][-1]
        tscale = max(rms(truth["y"]), 1e-300)
        for v in VARIANTS:
            r1 = None
            for m in ms:
                G = refine_grid(base, m, path, mask)
                out = run_scan(params, G.times, G.values, v, tau)
                r = readouts(G.times, out["y"], out["integral"])
                if m == 1:
                    r1 = r
                rec = {"exp": "FR-E3-READOUT", "config": tag, "seed": seed, "scenario": "b_newobs_nonuniform",
                       "variant": v, "m": m, "n_steps": G.n_steps, "scale": tscale, "readouts": r,
                       "true_avg": true_avg, "true_last": true_last}
                rec["err_truth_last"] = abs(r["last"] - true_last) / tscale
                for name in ("sample_mean", "time_riemann", "time_trapz", "time_exact"):
                    val = r[name]
                    rec[f"err_truth_{name}"] = None if val is None else abs(val - true_avg) / tscale
                rec["change_sample_sum"] = abs(r["sample_sum"] - r1["sample_sum"]) / (tscale * K1)
                recs.append(rec)
    return recs


# ----------------------------------------------------------------------------- E4

def _scalar_params(lam, b=1.0, c=1.0):
    # w = beta = 0  ->  g(u) = softplus(0) = ln 2 for every input
    return ToyParams((complex(lam, 0.0),), 0.0, 0.0, (complex(b, 0.0),), (0j,), (complex(c, 0.0),), (0j,))


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _fp32_zoh_scan(lam, g, b, u, dt, n, naive):
    """Emulated float32 exact-ZOH recurrence (real scalar mode).

    Every primitive result is rounded to float32; exp/expm1 are evaluated in
    float64 and rounded (i.e. a correctly rounded float32 exp).  This is an
    emulation, not a reproduction of any GPU kernel.
    """
    lam32, g32, b32, u32, dt32 = _f32(lam), _f32(g), _f32(b), _f32(u), _f32(dt)
    delta = _f32(dt32 * g32)
    z = _f32(delta * lam32)
    a = _f32(math.exp(z))
    em1 = _f32(a - 1.0) if naive else _f32(math.expm1(z))
    bb = _f32(_f32(em1 / lam32) * b32)
    h = 0.0
    for _ in range(n):
        h = _f32(_f32(a * h) + _f32(bb * u32))
    return h


def run_e4(cfg):
    recs = []
    getcontext().prec = 50
    ln2 = math.log(2.0)

    # N1: cancellation in the ZOH input coefficient (exp(z)-1)/lambda for tiny z
    lam = -1.0
    for e in range(2, 15, 2):
        z = -(10.0 ** -e)
        exact = (Decimal(z).exp() - 1) / Decimal(lam)
        naive = (math.exp(z) - 1.0) / lam
        good = math.expm1(z) / lam
        recs.append({"exp": "FR-E4-NUMERICS", "case": "N1_cancellation", "z": z,
                     "rel_err_naive": float(abs((Decimal(naive) - exact) / exact)),
                     "rel_err_expm1": float(abs((Decimal(good) - exact) / exact))})

    # N1b: split [0,1] into m = 2^j held sub-steps, float64, naive vs expm1.
    # lambda=-1, g=ln2, B=C=u=1 -> exact h(1) = 0.5
    p = _scalar_params(-1.0)
    for j in range(0, 17, 2):
        m = 2 ** j
        times = [i / m for i in range(m + 1)]
        vals = [1.0] * m
        for naive in (False, True):
            out = run_scan(p, times, vals, "zoh_dt", 1.0, naive=naive)
            recs.append({"exp": "FR-E4-NUMERICS", "case": "N1b_split_drift_fp64",
                         "formula": "naive" if naive else "expm1", "m": m,
                         "rel_err_h1": abs(out["h"][-1][0].real - 0.5) / 0.5})

    # N2: bilinear on stiff modes (sign-alternating Abar) vs exact ZOH
    T, K = cfg["grids"]["T"], cfg["grids"]["K_base"]
    path = lambda t: math.sin(2.0 * t) + 0.5
    base = base_grid(base_times(T, K), path)
    for lam in (-5.0, -50.0, -500.0):
        p = _scalar_params(lam)
        exact = run_scan(p, base.times, base.values, "zoh_dt", T / K)["y"]
        for m in (1, 2, 4, 8, 16, 32):
            G = split_grid(base, m)
            for v in ("bilinear_dt", "eulerB_dt"):
                out = run_scan(p, G.times, G.values, v, T / K)
                z = (T / K / m) * ln2 * lam
                abar = (1 + z / 2) / (1 - z / 2) if v.startswith("bilinear") else math.exp(z)
                recs.append({"exp": "FR-E4-NUMERICS", "case": "N2_stiff", "lambda": lam, "variant": v,
                             "m": m, "abar": abar,
                             "artifact_y": rel_l2(at_index(out["y"], G.base_index), exact)})

    # N3: Euler-B on coarse steps: steady state under held u=1 (exact h* = 1)
    p = _scalar_params(-1.0)
    for dt in (0.01, 0.1, 0.5, 2.0, 8.0):
        n = int(math.ceil(60.0 / (dt * ln2))) + 5
        times = [i * dt for i in range(n + 1)]
        for v in ("zoh_dt", "eulerB_dt", "bilinear_dt"):
            out = run_scan(p, times, [1.0] * n, v, dt)
            recs.append({"exp": "FR-E4-NUMERICS", "case": "N3_coarse_steady_state", "dt": dt, "variant": v,
                         "steady_state": out["h"][-1][0].real, "rel_err": abs(out["h"][-1][0].real - 1.0)})

    # N4: emulated float32 composition drift, split [0,1] into m = 2^j
    for j in range(0, 15, 2):
        m = 2 ** j
        for naive in (False, True):
            h = _fp32_zoh_scan(-1.0, ln2, 1.0, 1.0, 1.0 / m, m, naive)
            recs.append({"exp": "FR-E4-NUMERICS", "case": "N4_split_drift_fp32_emulated",
                         "formula": "naive" if naive else "expm1", "m": m, "rel_err_h1": abs(h - 0.5) / 0.5})
    return recs
