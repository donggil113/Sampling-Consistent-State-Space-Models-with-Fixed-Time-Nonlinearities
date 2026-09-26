"""Evaluate the pre-registered FIRST_RUN hypotheses on the primary configuration.

Only the primary configuration (uniform base grid, real modes, 32 seeds) is
used; secondary configurations are descriptive (amendment A3).  Missing
records yield NOT_RUN, never PASS.
"""

from .metrics import loglog_slope, median

PRIMARY = "uniform-real"


def _sel(recs, **kw):
    out = []
    for r in recs:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def _by_seed(recs, key):
    d = {}
    for r in recs:
        d.setdefault(r["seed"], {})[r["m"]] = r[key]
    return d


def _slopes(recs, key, ms):
    out = []
    for seed, row in sorted(_by_seed(recs, key).items()):
        if all(m in row for m in ms):
            out.append(loglog_slope(ms, [row[m] for m in ms]))
    return out


def _not_run(hid, why):
    return {"id": hid, "status": "NOT_RUN", "reason": why}


def _res(hid, ok, kind, evidence):
    return {"id": hid, "status": "PASS" if ok else "FAIL", "kind": kind, "evidence": evidence}


def evaluate(e1, e2, e3):
    H = []
    E1 = _sel(e1 or [], config=PRIMARY, kind="variant")
    E2 = _sel(e2 or [], config=PRIMARY, kind="variant")
    E3a = _sel(e3 or [], config=PRIMARY, scenario="a_split_nonuniform")
    E3b = _sel(e3 or [], config=PRIMARY, scenario="b_newobs_nonuniform")

    # H1
    r = [x for x in _sel(E1, variant="zoh_dt") if x["m"] in (2, 4, 8)]
    if r:
        mx = max(x["change_y"] for x in r)
        mxh = max(x["change_h"] for x in r)
        H.append(_res("H1", mx <= 1e-12, "engineering check of a known identity",
                      {"max_change_y": mx, "max_change_h": mxh, "threshold": 1e-12, "n_records": len(r)}))
    else:
        H.append(_not_run("H1", "no FR-E1 zoh_dt records"))

    # H2, H3
    for hid, v, lo, hi in (("H2", "eulerB_dt", -1.2, -0.8), ("H3", "bilinear_dt", -2.2, -1.8)):
        rr = _sel(E1, variant=v)
        if rr:
            s = _slopes(rr, "artifact_y", [1, 2, 4, 8])
            med = median(s)
            ok = med is not None and lo <= med <= hi
            H.append(_res(hid, ok, "known convergence order (numerical check, not a proof)",
                          {"median_slope": med, "range": [lo, hi], "min_slope": min(x for x in s if x is not None),
                           "max_slope": max(x for x in s if x is not None), "n_units": len(s)}))
        else:
            H.append(_not_run(hid, f"no FR-E1 {v} records"))

    # H4
    ev, ok_all = {}, True
    for v in ("zoh_nodt", "eulerB_nodt"):
        rr = _sel(E1, variant=v)
        if not rr:
            H.append(_not_run("H4", f"no FR-E1 {v} records"))
            break
        row = _by_seed(rr, "artifact_y")
        frac = sum(1 for s in row.values() if s[8] >= s[2]) / len(row)
        med8 = median([s[8] for s in row.values()])
        ok = frac >= 0.9 and med8 >= 0.1
        ok_all &= ok
        ev[v] = {"frac_nondecreasing_m2_to_m8": frac, "median_artifact_m8": med8, "ok": ok}
    else:
        H.append(_res("H4", ok_all, "toy constructed to be grid-dependent (demonstration, not discovery)", ev))

    # H5
    rr = _sel(E2, variant="zoh_dt")
    if rr:
        mx = max(x["artifact"] for x in rr)
        s = _slopes(rr, "err_to_truth", [1, 2, 4, 8])
        med = median(s)
        ok_a = mx <= 1e-8
        ok_b = med is not None and -1.2 <= med <= -0.8
        H.append(_res("H5a", ok_a, "engineering check (exact ZOH vs independent RK4 on held path)",
                      {"max_artifact": mx, "threshold": 1e-8}))
        H.append(_res("H5b", ok_b, "known first-order hold convergence (numerical check)",
                      {"median_slope_err_to_truth": med, "range": [-1.2, -0.8],
                       "min_slope": min(x for x in s if x is not None), "max_slope": max(x for x in s if x is not None)}))
    else:
        H.append(_not_run("H5a", "no FR-E2 zoh_dt records"))
        H.append(_not_run("H5b", "no FR-E2 zoh_dt records"))

    # H6
    ev, ok_all, missing = {}, True, False
    for v in ("zoh_nodt", "eulerB_nodt"):
        rr = [x for x in _sel(E2, variant=v) if x["m"] == 8]
        if not rr:
            missing = True
            break
        med8 = median([x["artifact"] for x in rr])
        ok_all &= med8 >= 0.1
        ev[v] = {"median_artifact_m8": med8}
    H.append(_not_run("H6", "missing FR-E2 nodt records") if missing else
             _res("H6", ok_all, "toy constructed to be grid-dependent (demonstration)", ev))

    # H7
    rr = _sel(E3a, variant="zoh_dt")
    if rr:
        mx = max(x["change_time_exact"] for x in rr if x["m"] in (2, 4, 8))
        H.append(_res("H7a", mx <= 1e-12, "engineering check (additivity of exact integral)",
                      {"max_change_time_exact": mx, "threshold": 1e-12}))
        ev, ok = {}, True
        for m in (2, 4, 8):
            ms = [x for x in rr if x["m"] == m]
            s_sum = median([x["change_sample_sum"] for x in ms])
            s_mean = median([x["change_sample_mean"] for x in ms])
            ev[f"m{m}"] = {"median_change_sample_sum": s_sum, "median_change_sample_mean": s_mean}
            ok &= s_sum >= 1e-2 and s_mean >= 1e-2
        ev["operationalization"] = "threshold 1e-2 required at every m in {2,4,8}"
        H.append(_res("H7b", ok, "readout property (known in principle)", ev))
        s = _slopes(rr, "change_time_riemann", [2, 4, 8])
        med = median(s)
        pos = all(median([x["change_time_riemann"] for x in rr if x["m"] == m]) > 0 for m in (2, 4, 8))
        H.append(_res("H7c_original", pos and med is not None and med < 0,
                      "MIS-SPECIFIED pre-run (see amendment A2); evaluated verbatim",
                      {"median_slope_change_time_riemann_m2_m8": med, "all_medians_positive": pos}))
        s2 = _slopes(rr, "riemann_minus_exact", [1, 2, 4, 8])
        med2 = median(s2)
        H.append(_res("H7c_restated", med2 is not None and med2 < 0, "known Riemann-sum convergence (numerical check)",
                      {"median_slope_riemann_minus_exact": med2}))
    else:
        for hid in ("H7a", "H7b", "H7c_original", "H7c_restated"):
            H.append(_not_run(hid, "no FR-E3(a) zoh_dt records"))

    # H8
    rr = [x for x in _sel(E3b, variant="zoh_dt") if x["m"] == 8]
    if rr:
        sm = median([x["err_truth_sample_mean"] for x in rr])
        te = median([x["err_truth_time_exact"] for x in rr])
        H.append(_res("H8", sm >= 5 * te, "readout property under density change",
                      {"median_err_sample_mean_m8": sm, "median_err_time_exact_m8": te, "ratio": sm / te if te else None}))
    else:
        H.append(_not_run("H8", "no FR-E3(b) zoh_dt records"))

    # S2 research stop condition
    st = {h["id"]: h["status"] for h in H}
    if st.get("H1") == "PASS" and st.get("H5a") == "PASS":
        s2 = "TRIGGERED: simple time handling (Delta=dt*g, exact ZOH, time-exact readout) removes the toy's splitting "\
             "artifact to float64 precision and its new-observation artifact to reference precision -> "\
             "ARCHITECTURE_CLAIM=NOT_SUPPORTED_BY_TOY"
    elif "NOT_RUN" in (st.get("H1"), st.get("H5a")):
        s2 = "NOT_EVALUATED"
    else:
        s2 = "NOT_TRIGGERED (H1 or H5a failed -> treat as bug per S1 before interpreting)"
    return {"hypotheses": H, "stop_condition_S2": s2}
