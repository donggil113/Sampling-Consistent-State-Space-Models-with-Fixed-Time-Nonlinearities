# n1ref — vendored reference code from project N1

Verbatim copy of `src/scssm/*.py` from
`donggil113/Sampling-Consistent-Selective-State-Space-Models` at commit
`306a7727de1c312629f29d8ddaebcf797e3b31da` (branch `claude/intelligent-maxwell-t1lcda`).
The only edit is the package name in import lines (`scssm.` -> `n1ref.`).

It is pure CPython (stdlib only, float64 / complex128) and is used here as an
**independent reference** for the torch implementation in `src/fxclock`:

- `model.cexpm1`, `model.phi1`, `model.phi1m1`: small-argument-safe exponential functions.
- `model.run_scan(..., "zoh_dt")`: exact ZOH recurrence on a right-held grid.
- `reference.ct_held`: classical-RK4 reference on the held path (does not use ZOH formulas).
- `experiments._fp32_zoh_scan`: emulated float32 composition (N1 check FR-E4-N4).
- `grids.split_grid`: interval splitting with repeated held values.
