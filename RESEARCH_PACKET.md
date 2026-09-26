# Research packet — N2: Sampling-consistent SSMs with fixed-time nonlinearities

Last updated: 2026-09-26.

- Pre-registration: `configs/prereg_n2.json`, commit `2a29f83`, before any test-set evaluation.
- Deviations: `docs/DEVIATIONS.md`.
- Theory ledger: `docs/THEORY.md`.
- Literature: `RELATED_WORK.md`.

## 0. Verdict

<!--VERDICT-->

## 1. Question and scope

**Question.** When only the time grid of the *same reconstructed input* changes, does a nonlinear update at every observation make the representation grid-dependent? Does a model that separates exact linear input integration from nonlinear updates on a fixed physical-time clock add anything beyond fixed-grid resampling and prior work?

**Evidence used**

1. **Analytic and float64 counterexamples.** `scripts/counterexample.py` (CE1–CE3), independent of the main code.
2. **Numerical checks of the implementation.** 23 unittest checks; `results/numerics.json`; GPU gate.
3. **A released foundation model.** FlowState (ibm-granite, Apache-2.0) on real ETTm1 data (FS1–FS3).
4. **A trained real-signal task.** UCI-HAR, 50 Hz, 9 channels, subject-disjoint test set. Baselines and a 2×2 ablation, 3 seeds, cost including preprocessing.

**Independent unit.** In HAR the unit is the test subject (9). Windows overlap by 50% and are not independent. Seeds are reported separately. In FS1 the unit is the context window (16), and windows from one series are not independent.

## 2. Grid-change taxonomy and published rules

Code: `src/fxclock/knots.py`, `src/fxclock/prep.py`, `src/fxclock/conditions.py`.

**Knots.** Every knot carries a `measured` flag. Virtual knots are placed *on* a published reconstruction and are never counted as measurements. Downsampling and dropping act only on measured knots.

| Change | Definition | Reconstructed path | Conditions |
|---|---|---|---|
| Lossless knot refinement (FOH) | virtual knots on the linear interpolant | unchanged for FOH | `foh_m2`, `foh_m4` (uniform), `foh_rand` (127 random) |
| Lossless knot refinement (ZOH) | virtual knots carrying the right-held value | unchanged for right-hold ZOH (the hold implied by h_k = Ā(dt_k)h_{k−1} + B̄(dt_k)u_k) | `zoh_m2` |
| Real downsampling | every M-th measured knot | changes; aliasing possible, not recoverable (`docs/THEORY.md` §0) | `down2` (25 Hz), `down4` (12.5 Hz) |
| Missing observations | measured interior knots dropped independently (first and last kept) | changes | `drop30`, `drop50`, `drop70` |
| Re-knotted twins | same path as `down2` / `drop50`, plus virtual knots at the native times | same as the lossy parent | `down2_reknot`, `drop50_reknot` |

A lossless refinement is lossless only for the rule used to place the knots. Each model is tested under the refinement that is lossless for *its own* input rule, and under the other one.

**Reconstruction.**

- **FOH:** linear interpolation, with constant hold outside the knots.
- **ZOH:** right hold.

**Clock.** τ_j = j·Δ_c with Δ_c ∈ {0.02 s, 0.08 s}. It does not depend on the observations, and τ_j equals the native knot times bit-for-bit (`clock_times`).

**Output query rules.**

- **Clocked models.** y_j at τ_j depends only on X on [0, τ_j]. The readout is the uniform mean over j. In real time, y_j can be emitted once the first measurement at or after τ_j has arrived, so the latency is at most one observation gap.
- **Per-observation models.** y_k at every knot. The readout is time-weighted with weights (t_k − t_{k−1})/T. A sample-count mean is avoided because it is grid-dependent by construction (N1 C5).
- **NRDE.** State after the last clock window.

## 3. Minimal counterexample (answers the core question at the smallest scale)

`results/counterexample.json`. CE1 is exact rational arithmetic; CE2 and CE3 are float64 with 8 random units.

| Case | Grid change | Result |
|---|---|---|
| CE1: x(t)=t on [0,1], σ(u)=u² at each knot, then the exact integral of the FOH re-reconstruction | add a virtual knot at ½ on the line | **1/2 → 3/8** (continuous limit 1/3). With ZOH re-reconstruction: 1 → 5/8. σ on a fixed clock {0,½,1}: 3/8 for both. Affine σ: 5/2 = 5/2 |
| CE2: 2-layer S5/FlowState-style stack (exact ZOH, physical dt, GELU at every knot, right-hold re-reconstruction) | split every held interval into m pieces with the same held value (lossless for ZOH) | layer 1: ≤ 6.8e-15 (exact). Layer-2 output: median relative change **0.19 / 0.28 / 0.33 / 0.35** at m = 2/4/8/16 |
| CE2, σ = identity (linear stack, per-knot re-hold) | same | **0.22 / 0.32 / 0.37 / 0.40**. The change comes from per-knot re-sampling of intermediate signals, even without a nonlinearity |
| CE2, exact joint integration of the linear cascade | same | ≤ 6.1e-15 |
| CE2, GELU only on a fixed clock | same | ≤ 8.0e-15 |
| CE3: one selective layer (Mamba-like gate per knot, Δ = dt·g(u), exact ZOH) | ZOH split / FOH virtual knots | ZOH split ≤ 2.3e-15 (reproduces N1 H1). FOH virtual knots: **0.19 / 0.28 / 0.31** at m = 2/4/8 |

**Reading.**

- A per-observation nonlinearity forces intermediate signals to be sampled at the observation knots. That makes the representation grid-dependent even when every linear part is exact and dt-scaled.
- For a linear stack, the same dependence can be removed by exact cascade integration. For a nonlinear stack it cannot in closed form.
- `docs/THEORY.md` S2 proves the FOH case: invariance for all inputs ⇔ σ affine. The principle is known in operator learning (SNO, ReNO, CNO; `RELATED_WORK.md`).

## 4. Minimal model and its numerical checks

**Model.** An exact linear input-integration stem: a diagonal complex SSM integrated in closed form over the FOH path, using the window-sum formulation of `src/fxclock/stem.py`. Its states z_j = [Re h(τ_j), Im h(τ_j), X(τ_j)] are read at the fixed clock. A standard S4D-style deep SSM (GELU + GLU, LayerNorm) runs on the clock, and this is the only place nonlinear updates happen. The ablation replaces the stem by per-observation ZOH on held values, and/or runs the same backbone per observation with dt_k.

**Checks.** `python3 -m unittest tests.test_numerics`: 23/23 pass. The N1 checks are reused through the vendored `src/n1ref`.

| Check | Result |
|---|---|
| φ1, φ2 vs mpmath (60 digits), \|z\| from 1e-14 to 3e2, 7 angles | ≤ 2e-14 relative (float64) |
| ZOH stem vs N1 exact ZOH (`n1ref.run_scan zoh_dt`) on N1's jittered grid | ≤ 1e-13 |
| ZOH stem vs N1 independent RK4 (`n1ref.ct_held`) | ≤ 1e-9 |
| FOH stem vs an independent RK4 on the FOH path | ≤ 1e-10 |
| N1 `split_grid` ≡ ZOH-virtual refinement; stem invariant | ≤ 1e-13 |
| FOH-virtual refinement (m = 2, 8, 500 random) | ≤ 1e-13. A ZOH-valued knot changes it by > 1e-4, so the test can fail |
| Extreme refinement m = 1000 (segments 2e-5 s, series branch of φ2 exercised) | ≤ 1e-12 |
| Every path-functional model is exactly invariant under FOH refinement; every per-observation model changes | path functionals ≤ 1e-12; per-observation > 1e-4 |
| Padding inert (short and long windows in one batch) | ≤ 1e-12 |
| Depth-2 log-signature vs brute-force Riemann–Stieltjes; refinement invariance; Chen global = direct | ≤ 1e-9 / 1e-13 / 1e-12 |
| Bilinear stem vs a sequential Tustin reference | ≤ 1e-13 |
| gradcheck: φ across the series/direct branch; stem w.r.t. inputs, Λ and B; custom scan adjoint | pass |
| Parameter gradients under refinement (P1, P4) | ≤ 1e-11 relative. A1 (per-observation) > 1e-4 |
| Scan (sequential / Hillis–Steele / custom adjoint) vs reference; LTI FFT vs scan | ≤ 1e-13; ≤ 1e-10 |

**Small-dt stability and floating-point floors** (`results/numerics.json`; real HAR windows, 32 modes, clock 0.08 s). Values are the median relative change of the stem state at the clock vs the unrefined grid.

| m (virtual knots per interval) | min \|λd\| | float64 window-sum | float32 window-sum | float32 window-sum, naive φ | float32 sequential composition |
|---|---|---|---|---|---|
| 1 | 5.9e-2 | 0 | 2.5e-7 | 2.7e-7 | 2.4e-7 |
| 4 | 1.5e-2 | 2.9e-16 | 2.4e-7 | 4.3e-6 | 3.6e-6 |
| 64 | 9.3e-4 | 6.0e-16 | 3.7e-7 | 2.9e-5 | 2.3e-5 |
| 1024 | 5.8e-5 | 1.9e-15 | 1.5e-6 | 3.2e-5 | 6.5e-5 |
| 4096 | 1.5e-5 | 4.8e-15 | 8.7e-6 | 1.2e-3 | — |

- Naive φ2 in float32 has relative error 4.1 at |z| = 1e-4, and it is unusable for refinement.
- The stable φ with the window-sum keeps the float32 floor ≤ 1.6e-5 up to m = 4096.

**GPU gate** (`scripts/gpu_gate.py`). No GPU gate existed in the accessible prior repositories; N1 recorded `"gpu": false` only. A new gate compares CUDA float32 against CPU float64. It records **NOT_RUN** because this container has no CUDA device (`results/gpu_gate.json`).

## 5. Real published model: FlowState (FS1–FS3)

Scripts `scripts/flowstate_probe.py` (FS1), `flowstate_attribution.py` (FS2) and `flowstate_fp64_layers.py` (FS3); table `results/flowstate_summary.md`.

**Setup.** ETTm1 column OT at 15-minute sampling, which is FlowState's own benchmark family. There are 16 context windows ending in the standard test period, with context L = 1024 and horizon H = 96. The scale factor is s = `get_fixed_factor("15T")` = 0.25. Refinement by r uses s/r; downsampling by r uses s·r. This follows the paper's rule. Forecasts are compared at common physical future times. Repeating the native run gives an exactly identical output (change 0).

**The test that matches FlowState's own claim is ZOH repeat.** Every sample is repeated r times. That is lossless for the kernel x_k = Ā x_{k−1} + B̄ u_k, including the implicit one-step hold of the first sample (deviation D4). Under it the first S5 layer is exact by FlowState App. B.1, which says the pointwise operations "do not introduce additional temporal discretization error".

FS1. Relative change of the median forecast at common physical times, and of the last encoder state, vs native. Median [min, max] over windows. zohR = every sample repeated r times, lossless for FlowState's own right-hold ZOH kernel; refine = FOH virtual knots; down = every r-th real sample (lossy).

| revision | params | grid change | lossless for FlowState's rule? | forecast change | state change |
|---|---|---|---|---|---|
| v1.0 (main) | 9.1M | ZOH repeat x2 | yes | 4.2% [1.2, 14.3] | 31.8% [17.3, 44.1] |
| v1.0 (main) | 9.1M | ZOH repeat x4 | yes | 4.2% [1.3, 15.0] | 33.3% [18.1, 46.6] |
| v1.0 (main) | 9.1M | FOH virtual knots x2 | no (FOH path) | 4.7% [1.4, 15.2] | 34.3% [19.1, 49.6] |
| v1.0 (main) | 9.1M | FOH virtual knots x4 | no (FOH path) | 4.8% [1.3, 15.1] | 36.8% [20.7, 55.3] |
| v1.0 (main) | 9.1M | real downsampling /2 | no (lossy) | 2.2% [1.0, 12.2] | 22.2% [16.2, 42.6] |
| v1.0 (main) | 9.1M | real downsampling /4 | no (lossy) | 4.8% [2.2, 16.9] | 54.3% [37.3, 70.5] |
| v1.0 (main) | | repeat of native input | yes | 0.0e+00 | |

v1.0 (main): forecast MAE native 0.678 vs FOH-refined x2 0.741 (median over windows).

| r1.1 | 18.5M | ZOH repeat x2 | yes | 7.0% [1.3, 23.9] | 45.3% [23.7, 63.7] |
| r1.1 | 18.5M | ZOH repeat x4 | yes | 6.9% [1.6, 20.6] | 43.0% [24.4, 64.8] |
| r1.1 | 18.5M | FOH virtual knots x2 | no (FOH path) | 6.7% [1.5, 19.9] | 36.9% [25.8, 58.2] |
| r1.1 | 18.5M | FOH virtual knots x4 | no (FOH path) | 6.7% [1.6, 18.8] | 39.8% [32.3, 59.8] |
| r1.1 | 18.5M | real downsampling /2 | no (lossy) | 3.9% [0.7, 13.7] | 38.5% [29.5, 52.1] |
| r1.1 | 18.5M | real downsampling /4 | no (lossy) | 9.4% [2.3, 19.5] | 74.8% [63.3, 98.9] |
| r1.1 | | repeat of native input | yes | 0.0e+00 | |

r1.1: forecast MAE native 0.988 vs FOH-refined x2 0.859 (median over windows).

FS2. Attribution on 8 windows: causal RevIN (running sample-count mean/std) vs fixed per-context statistics (affine, grid-independent). Per-layer relative change of the encoder output at the last context step.

| revision | RevIN | grid change | forecast change | per-layer change at last step (layer 0 -> 5) |
|---|---|---|---|---|
| v1.0 (main) | causal_revin | zohR2 | 6.8% | 0.2% 2.7% 8.6% 17% 27% 34% |
| v1.0 (main) | causal_revin | foh2 | 6.6% | 5.1% 7.4% 9.7% 19% 29% 36% |
| v1.0 (main) | fixed_revin | zohR2 | 5.1% | 0.002% 2.2% 8.8% 16% 24% 27% |
| v1.0 (main) | fixed_revin | foh2 | 5.2% | 4.9% 7.1% 9.4% 19% 24% 28% |
| r1.1 | causal_revin | zohR2 | 11.7% | 0.096% 1.3% 4.7% 37% 51% 46% |
| r1.1 | causal_revin | foh2 | 9.0% | 1.4% 5% 8.3% 31% 44% 40% |
| r1.1 | fixed_revin | zohR2 | 12.4% | 0.0038% 1.2% 4.1% 34% 47% 43% |
| r1.1 | fixed_revin | foh2 | 10.9% | 1.5% 4.7% 7.9% 26% 38% 36% |

FS3. Encoder in float32 vs float64 (weights cast), ZOH repeat x2, fixed RevIN statistics, 4 windows: median per-layer relative change at the last context step.

| revision | dtype | layer 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| v1.0 (main) | float32 | 1.9e-05 | 2.4e-02 | 9.1e-02 | 1.8e-01 | 2.6e-01 | 3.0e-01 |
| v1.0 (main) | float64 | 5.7e-14 | 2.4e-02 | 9.1e-02 | 1.8e-01 | 2.6e-01 | 3.0e-01 |
| r1.1 | float32 | 9.9e-06 | 2.0e-02 | 6.9e-02 | 3.2e-01 | 4.9e-01 | 4.9e-01 |
| r1.1 | float64 | 4.5e-14 | 2.0e-02 | 6.9e-02 | 3.2e-01 | 4.9e-01 | 4.9e-01 |


**Reading.**

1. **The forecast changes.** Under a refinement that is lossless for FlowState's own reconstruction rule, the released models change their forecast by a median 4.2% (v1.0) and 7.0% (r1.1), with a maximum of 24%. The last encoder state changes by 32–45%. This is *larger* than the change under real 2× downsampling (2.2% / 3.9%), where half the measurements are removed.
2. **Where it comes from (FS3).** In float64, layer 0 is exactly invariant (5.7e-14 / 4.5e-14), and layers 1–5 change by 2%–49%. The values are identical in float32 and float64, so this is not rounding. It is the CE2 mechanism: the per-sample outputs of layer l are re-held as the input of layer l+1.
3. **Not the normalisation (FS2).** Replacing the causal (sample-count) RevIN statistics with fixed ones leaves the effect in place: 5.1% / 12.4% forecast change.
4. **Accuracy moves in both directions.** Forecast accuracy is not systematically hurt or helped by refinement. MAE goes 0.678 → 0.741 for v1.0 and 0.988 → 0.859 for r1.1. The representation depends on the grid; that dependence is not an error signal.
5. **Scope.** This is a descriptive probe of one series and 16 windows, with inference only. FlowState's paper claims only *approximate* equivariance, with an error that vanishes as Δ → 0. The probe does not contradict that bound. It contradicts the App. B.1 sentence and quantifies the approximation at the native rate.


## 6. Real-signal task: UCI-HAR (pre-registered main arm)

<!--HAR-->

## 7. Decision (pre-registered rule)

<!--DECISION-->

## 8. Claims ledger

<!--LEDGER-->

## 9. Threats to validity and limits

<!--LIMITS-->

## 10. Sources, licenses, versions

| Item | Source / version | License | Use |
|---|---|---|---|
| This repository | `src/fxclock`, `scripts`, `tests`; CPython 3.11.15, torch 2.14.0+cpu, numpy 2.4.6, scipy 1.17.1, mpmath 1.3.0 | repository license NOT SPECIFIED (owner's decision) | executed |
| `src/n1ref` | N1 repository `donggil113/Sampling-Consistent-Selective-State-Space-Models` @ `306a772` (vendored) | same owner | executed as a reference |
| UCI-HAR | UCI ML Repository #240, zip sha256 `c00b8030…81031` | CC BY 4.0 | executed |
| ETTm1 | github.com/zhouhaoyi/ETDataset `ETT-small/ETTm1.csv`, sha256 `6ce1759b…6c9e` | CC BY-ND 4.0 (LICENSE file of the ETDataset repository, checked 2026-09-26); the data are used unmodified and not redistributed | executed (FS1–FS3) |
| FlowState | HF `ibm-granite/granite-timeseries-flowstate-r1` (main = v1.0, and revision r1.1); code `granite-tsfm` 0.3.9 (pip, `--no-deps`; its torch pin <2.12 was overridden, and it ran with torch 2.14) | Apache-2.0 | executed, inference only |
| Papers | `RELATED_WORK.md` | not checked | read |
