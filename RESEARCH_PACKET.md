# Research packet — N2: Sampling-consistent SSMs with fixed-time nonlinearities

Last updated: 2026-09-26.

- Pre-registration: `configs/prereg_n2.json`, commit `2a29f83`, before any test-set evaluation.
- Deviations: `docs/DEVIATIONS.md`.
- Theory ledger: `docs/THEORY.md`.
- Literature: `RELATED_WORK.md`.

## 0. Verdict

**NO_GO for any new-architecture claim.** Both parts of the pre-registered decision rule fail.

1. **G1: the gain is explained by simple resampling.**
   - The proposed model does not beat fixed-grid resampling + the same SSM on UCI-HAR.
     - P1 − point resampling: lossy mean −0.64 pp [−1.23, +0.00], native −0.41 pp.
     - P4 − point / box-filter resampling at the same clock: −0.35 / −0.68 pp.
   - It is also 2.0–3.2× more expensive per window, including preprocessing.
   - The one robust benefit of the fixed clock needs no exact stem. It is better robustness to heavy missingness: +0.5 to +0.8 pp lossy mean and +1.9 to +2.5 pp at 70% drop, with the same trained weights. The mandatory resampling baseline already delivers it.
2. **G2: prior work.** More than 20 sources have "substantial" overlap after adversarial re-checking (`RELATED_WORK.md`). They include:
   - Walker et al. 2026, with an explicit exact invariance to value-preserving insertions and a query partition independent of the observations;
   - Rough Transformer; Logsig-RNN; NRDE / Log-NCDE;
   - the multirate PDM-speech SSM 2026;
   - SFI audio layers;
   - SNO / ReNO / CNO.

**Core question, answered separately: yes, with a setting-dependent size.**

- Minimal counterexample: an exact change 1/2 → 3/8.
- A 2-layer FlowState-style stack: 19–35%.
- The released FlowState on real data, under a refinement that is lossless for its own rule: 4.2% / 7.0% median forecast change. Layer 0 is exact in float64 and deeper layers change by 2–49%.
- Trained HAR per-observation models: a real but practically negligible effect. Same-physical-time logit change is 1.5–3.5e-3 and flip rate ≤ 0.21%.
- Under real downsampling and missing observations, 50–114% of a per-observation model's output change is a knot artifact (same path, different knots). For every path functional it is 0.

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

All numbers: `results/har_summary.md` and `results/har_summary.json`, generated by `scripts/analyze_har.py`.

- **Training.** Native 50 Hz grid only. 15 epochs, AdamW, one shared setting per family, best dev epoch. 3 seeds.
- **Test.** 2947 windows from 9 subjects, disjoint from training.
- **Uncertainty.** Accuracy is the mean over seeds. The 95% CI is a cluster bootstrap over test subjects.
- **Cost.** Microseconds per window, preprocessing from raw knots included. Measured on an idle machine, 1 thread, 512 windows, median of 3 repeats (`results/cost_har.json`).

### 6.1 Accuracy, all models and conditions

| model | role | native | lossless (foh_m4) | down2 | down4 | drop30 | drop50 | drop70 | lossy mean | cost µs/window (native, incl. prep) |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_foh_clock | **proposed** (exact FOH stem + fixed clock, Δc = 0.02 s) | 92.5 [86.0, 97.5] | 92.5 | 92.4 | 92.4 | 92.4 | 92.0 | 90.1 | 91.9 | 2666 |
| A1_foh_perobs | ablation: exact stem only (same weights as P1) | 92.5 [86.0, 97.5] | 92.5 | 92.4 | 91.8 | 92.2 | 91.4 | 87.6 | 91.1 | 2830 |
| A2_zoh_clock | ablation: fixed clock only | 92.4 [85.7, 97.5] | 92.4 | 92.3 | 92.1 | 92.3 | 91.6 | 89.5 | 91.6 | 2320 |
| A3_zoh_perobs | ablation: neither (same weights as A2) | 92.4 [85.7, 97.5] | 92.5 | 92.2 | 91.6 | 92.1 | 91.2 | 87.5 | 90.9 | 2818 |
| B_point_clock1 | **mandatory**: fixed-grid resampling + SSM | 92.9 [86.3, 97.9] | 92.9 | 92.9 | 92.9 | 92.9 | 92.6 | 91.2 | 92.5 | 1320 |
| B_dtonly | dt-only step-size rescaling (S5/FlowState-style; same weights as B_point_clock1) | 92.9 [86.3, 97.9] | 92.8 | 92.9 | 92.7 | 92.6 | 92.4 | 89.4 | 92.0 | 1775 |
| B_tf_clock1 | resampling + Transformer | 88.3 [80.7, 94.3] | 88.3 | 88.6 | 88.0 | 88.6 | 87.9 | 86.2 | 87.9 | 1419 |
| P4_foh_clock | **proposed**, coarse clock Δc = 0.08 s | 92.0 [86.1, 96.7] | 92.0 | 92.0 | 91.8 | 91.9 | 91.5 | 89.1 | 91.3 | 1487 |
| A2_zoh_clock4 | fixed clock only, coarse | 92.5 [86.1, 97.5] | 92.5 | 92.4 | 92.0 | 92.1 | 91.3 | 88.3 | 91.2 | 1317 |
| B_point_clock4 | **mandatory**, coarse: point resampling + SSM | 92.0 [85.5, 97.1] | 92.0 | 92.0 | 92.0 | 92.1 | 92.0 | 89.9 | 91.6 | 466 |
| B_binmean_clock4 | resampling with box prefilter + SSM | 92.5 [86.0, 97.4] | 92.5 | 92.5 | 92.3 | 92.4 | 92.1 | 90.3 | 91.9 | 640 |
| B_patch_clock4 | fine resampling + linear patch + SSM | 92.0 [86.9, 96.2] | 92.0 | 92.1 | 91.9 | 92.0 | 91.6 | 89.4 | 91.4 | 507 |
| B_tfpatch_clock4 | fine resampling + patch Transformer | 90.4 [83.0, 96.2] | 90.4 | 90.4 | 90.1 | 90.4 | 89.8 | 88.2 | 89.8 | 364 |
| B_nrde_clock4 | close work: NRDE-style log-ODE | 87.9 [80.7, 93.7] | 87.9 | 87.4 | 67.5 | 88.1 | 86.1 | 77.7 | 81.4 | 697 |
| B_rformer_clock4 | close work: RFormer-style | 89.4 [81.8, 95.3] | 89.4 | 89.5 | 77.4 | 89.0 | 86.6 | 79.2 | 84.3 | 896 |
| B_bilin_clock4 | close work: multirate-SSM-style bilinear stem | 91.7 [85.7, 96.4] | 91.8 | 91.7 | 91.6 | 91.7 | 91.3 | 89.2 | 91.1 | 780 |

**Notes on this table.**

- **Same weights.** On the native grid the pairs P1/A1, A2/A3 and B_point_clock1/B_dtonly are the same function, so they share trained weights (THEORY S7).
- **Coarse clock and downsampling.** For B_point_clock4 the clock ticks are exactly the samples kept by down2 and down4, so those two conditions leave its input unchanged. That is a property of point sampling, not an advantage of the model: it never used the other samples.
- **Close-work baselines.** The NRDE- and RFormer-style baselines are this project's implementations (depth-2 log-signatures, one Euler step per window, basepoint augmentation; deviation D1). They are weak under down4, where the per-window log-signature changes a lot. They are not the published models.

### 6.2 2×2 ablation (exact stem × fixed clock) and key contrasts

Paired over windows, seed-averaged, subject-bootstrap 95% CI, in percentage points.

| contrast | what it isolates | lossy mean | drop70 | native |
|---|---|---|---|---|
| P1 − A1 (same weights) | fixed clock vs per-observation, exact FOH stem | **+0.80 [+0.42, +1.20]** | **+2.50 [+1.66, +3.33]** | 0 (identical) |
| A2 − A3 (same weights) | fixed clock vs per-observation, ZOH stem | **+0.67 [+0.46, +0.88]** | **+2.06 [+1.47, +2.68]** | 0 |
| B_point_clock1 − B_dtonly (same weights) | fixed clock vs per-observation, no stem | **+0.49 [+0.13, +0.86]** | **+1.85 [+0.91, +2.84]** | 0 |
| P1 − A2 | exact FOH stem vs ZOH stem, both on the clock | +0.29 [−0.10, +0.73] | | +0.08 |
| P4 − A2_zoh_clock4 | same, coarse clock | +0.04 [−1.13, +1.26] | | −0.44 |
| P1 − B_point_clock1 | exact stem vs no stem (mandatory baseline) | −0.64 [−1.23, +0.00] | | −0.41 [−1.03, +0.23] |
| P4 − B_point_clock4 | same, coarse clock | −0.35 [−1.46, +0.72] | | 0.00 |
| P4 − B_binmean_clock4 | exact stem vs box prefilter | −0.68 [−1.84, +0.49] | | −0.51 |
| P4 − B_patch_clock4 | exact stem vs fine resampling + patch | −0.15 [−1.74, +1.44] | | 0.00 |
| P1 − B_dtonly | proposed vs dt-only step-size rescaling | −0.15 [−0.89, +0.57] | | −0.41 |
| P1 − B_bilin_clock4 / P4 − B_bilin_clock4 | exact vs bilinear stem (multirate-SSM-style) | +0.79 [−0.71, +2.18] / +0.18 [−0.44, +0.77] | | +0.78 / +0.34 |

**Reading.**

- **Fixed clock: helps.** The fixed physical-time clock for the nonlinear updates improves robustness to missing observations. This holds for every stem and on identical weights.
- **Exact stem: does not help.** The exact FOH stem adds nothing measurable over a ZOH stem or over no stem at all. The mandatory baseline, point resampling onto the clock, is the best model overall.

### 6.3 Consistency under lossless refinement (float32, full test set)

Each cell is the median relative logit change vs native / flip rate. The fp64 column is the maximum over 3 seeds and 256 windows.

| model | foh_m2 | foh_m4 | foh_rand | zoh_m2 | fp64 max (FOH refinements) | artifact share down2 / drop50 (rule-matched twin) |
|---|---|---|---|---|---|---|
| P1_foh_clock | 6.4e-08 / 0.00% | 6.3e-08 / 0.00% | 6.4e-08 / 0.00% | 2.6e-04 / 0.07% | 4.3e-16 | 0.00 / 0.00 |
| A1_foh_perobs | 3.3e-03 / 0.08% | 4.9e-03 / 0.10% | 2.1e-03 / 0.10% | 3.1e-03 / 0.06% | 4.0e-02 | 1.14 / 0.83 |
| A2_zoh_clock | 2.5e-04 / 0.01% | 3.8e-04 / 0.02% | 1.6e-04 / 0.01% | 6.4e-08 / 0.00% | 1.2e-02 | 0.00 / 0.00 |
| A3_zoh_perobs | 3.4e-03 / 0.10% | 5.0e-03 / 0.15% | 2.2e-03 / 0.07% | 3.1e-03 / 0.08% | 4.6e-02 | 0.82 / 0.51 |
| B_point_clock1 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 | 0.00 / 0.00 |
| B_dtonly | 4.3e-03 / 0.19% | 6.2e-03 / 0.21% | 2.6e-03 / 0.10% | 4.1e-03 / 0.09% | 3.3e-02 | 0.84 / 0.50 |
| B_tf_clock1 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 | 0.00 / 0.00 |
| P4_foh_clock | 7.0e-08 / 0.00% | 7.0e-08 / 0.00% | 7.0e-08 / 0.00% | 6.2e-04 / 0.08% | 5.0e-16 | 0.00 / 0.00 |
| A2_zoh_clock4 | 5.0e-04 / 0.09% | 7.4e-04 / 0.16% | 3.8e-04 / 0.05% | 7.0e-08 / 0.00% | 2.8e-02 | 0.00 / 0.00 |
| B_point_clock4 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 | 0.00 / 0.00 |
| B_binmean_clock4 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 1.7e-03 / 0.12% | 4.6e-16 | 0.00 / 0.00 |
| B_patch_clock4 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 | 0.00 / 0.00 |
| B_tfpatch_clock4 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 | 0.00 / 0.00 |
| B_nrde_clock4 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 5.1e-03 / 0.41% | 2.2e-14 | 0.00 / 0.00 |
| B_rformer_clock4 | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 0.0e+00 / 0.00% | 1.7e-03 / 0.29% | 1.4e-14 | 0.00 / 0.00 |
| B_bilin_clock4 | 2.4e-03 / 0.15% | 2.5e-03 / 0.18% | 2.3e-03 / 0.15% | 2.5e-03 / 0.18% | 1.1e-01 | 0.92 / 0.19 |

**Consistency findings.**

- **H1 (exactness): PASS.** Every path functional changes by ≤ 2.2e-14 in float64 and flips 0 predictions in float32. The float32 floor of the exact-stem models is 6–7e-8.
- **Exactness is shared by all path functionals.** Plain fixed-grid resampling is *exactly* invariant too (0.0), so exactness is not a contribution of the exact stem (THEORY S1).
- **H2 (per-observation dependence): numerically PASS, practically FAIL.** The per-observation models change by a median 2.1–6.2e-3 (float64: 2.6–5.6e-3), which is ≥ 1e-3 and ≈ 10⁵× the float32 floor. But flip rates are ≤ 0.21%, below the pre-registered 0.5% relevance bar, and accuracy changes by ≤ 0.1 pp.
- **Artifact share (rule-matched twins, D7).**
  - Per-observation models: 0.50–1.14 under down2 / drop50. Most of their output change under lossy grids is a knot artifact rather than information loss.
  - Every fixed-clock model, including A2 with the ZOH stem under its own rule: 0.

**H2 decomposition (post hoc, D7; `results/har_supplementary.json`).** The per-observation readout is a time-weighted Riemann sum over knots, so part of the change is quadrature. Median over windows, then over seeds, float32 full test set:

| model | condition | full | same physical times | readout control (grid-free) | float64 full / same times (256 windows) |
|---|---|---|---|---|---|
| A1_foh_perobs | foh_m2 | 3.3e-3 | 2.3e-3 | 1.3e-3 | 3.9e-3 / 2.5e-3 |
| A1_foh_perobs | foh_m4 | 4.9e-3 | 3.5e-3 | 1.9e-3 | 5.6e-3 / 4.0e-3 |
| A1_foh_perobs | foh_rand | 2.1e-3 | 1.5e-3 | 0.9e-3 | 2.6e-3 / 1.6e-3 |
| A3_zoh_perobs | zoh_m2 | 3.1e-3 | 1.8e-3 | 0 | 3.4e-3 / — |
| B_dtonly | zoh_m2 | 4.1e-3 | 1.7e-3 | 0 | 5.0e-3 / — |

The representation itself changes at the same physical times, by 1.5–3.5e-3, which is above the 1e-3 threshold. A readout-quadrature term of similar size comes on top. For right-hold models under ZOH refinement the control is 0, because a held grid-free signal pools identically.

### 6.4 Cost including preprocessing (µs per window, 1 thread)

| model | prep native | forward native | total native | total foh_m4 | total drop50 | params |
|---|---|---|---|---|---|---|
| P1_foh_clock | 194 | 2472 | 2666 | 7261 | 2444 | 56582 |
| A1_foh_perobs | 189 | 2642 | 2830 | 17208 | 28454 | 56582 |
| A2_zoh_clock | 122 | 2198 | 2320 | 4744 | 2484 | 56582 |
| A3_zoh_perobs | 121 | 2696 | 2818 | 16492 | 25589 | 56582 |
| B_point_clock1 | 92 | 1228 | 1320 | 1597 | 1288 | 51846 |
| B_dtonly | 86 | 1688 | 1775 | 12430 | 24816 | 51846 |
| B_tf_clock1 | 91 | 1328 | 1419 | 1610 | 1362 | 135046 |
| P4_foh_clock | 203 | 1284 | 1487 | 3991 | 1189 | 56582 |
| A2_zoh_clock4 | 121 | 1196 | 1317 | 3620 | 1110 | 56582 |
| B_point_clock4 | 78 | 388 | 466 | 670 | 506 | 51846 |
| B_binmean_clock4 | 242 | 398 | 640 | 1011 | 745 | 52422 |
| B_patch_clock4 | 118 | 390 | 507 | 716 | 531 | 53574 |
| B_tfpatch_clock4 | 134 | 230 | 364 | 580 | 382 | 136774 |
| B_nrde_clock4 | 510 | 188 | 697 | 1609 | 739 | 121254 |
| B_rformer_clock4 | 660 | 236 | 896 | 1870 | 864 | 141510 |
| B_bilin_clock4 | 210 | 570 | 780 | 1619 | 765 | 56582 |

**Cost findings.**

- The exact stem makes a clocked model 2.0× (Δc = 0.02 s) to 3.2× (Δc = 0.08 s) more expensive than point resampling at the same clock.
- The stem's cost scales with the number of segments: ×2.7 under foh_m4, while the backbone cost does not grow.
- Per-observation models scale with the number of knots, ×6–7 under foh_m4. On irregular grids (drop50) they are 10–14× slower than native, because the LTI/FFT path is no longer available and the time-varying scan is needed.
- Clocked models always run the LTI path.
- Training seconds per epoch are in `results/har_summary.json` (`epoch_seconds_median_contended`). They were measured under 4–6-process contention and are indicative only.

## 7. Decision (pre-registered rule)

Evaluated in code (`results/har_summary.json` → `decision`).

| Criterion | Result |
|---|---|
| **G1**: P1 or P4 beats *every* simple-resampling baseline within 1.5× its cost by ≥ 1.0 pp lossy mean (CI > 0) and ≥ −0.5 pp native | **FAIL.** P1 is below B_point_clock1 (−0.64) and B_binmean_clock4 (−0.07), is not ≥ 1 pp above B_point_clock4 (+0.25) or B_patch_clock4 (+0.46), and meets the G1 margin only against the two Transformer baselines. P4 is below B_point_clock1 (−1.24), B_point_clock4 (−0.35), B_binmean_clock4 (−0.68) and B_patch_clock4 (−0.15) |
| **G2 (literature)**: no prior work with "substantial" overlap | **FAIL.** More than 20 sources after adversarial re-check; five have "full" overlap on the fixed-time-nonlinearity component |
| **G2 (close work)**: NRDE-, RFormer- and bilinear-style baselines not within 1.0 pp | **FAIL.** The bilinear multirate-style stem is within 1.0 pp (P1 +0.79, P4 +0.18). P beats our NRDE/RFormer-style implementations by 7–10 pp, but these are not the published models |
| **Verdict** | **NO_GO.** The gain is explained by simple resampling and by prior work |

**What the NO_GO does not say.**

- It does not say that the fixed clock is useless. Nonlinear updates on a fixed physical clock are measurably more robust to missing observations than per-observation updates with dt rescaling (§6.2).
- It says that this is achieved as well, and more cheaply, by fixed-grid resampling. That construction is standard (IP-Nets, mTAN, HiSS, Hasegawa et al. 2021 for HAR; `RELATED_WORK.md`).

## 8. Claims ledger

| ID | Claim | Status | Evidence |
|---|---|---|---|
| C0 | Exact ZOH/FOH integration is invariant to interval splitting; stable φ-functions; Grönwall and interpolation bounds | STANDARD (K1–K4) | tests; `docs/THEORY.md` §1 |
| C1 | A model that factors through R(O) on a fixed clock is exactly refinement-invariant. This includes plain fixed-grid resampling | PROVED (elementary, low novelty; S1) | HAR H1: all path functionals ≤ 2.2e-14 (float64) |
| C2 | Per-knot σ with FOH re-reconstruction is refinement-invariant for all inputs ⇔ σ affine | PROVED (elementary; S2). The principle is known in operator learning (SNO/ReNO/CNO) | CE1: 1/2 → 3/8 |
| C3 | In per-observation stacks the grid dependence comes from per-knot re-sampling of intermediate signals. It is present even for σ = identity and removable by exact cascade integration only in the linear case | NUMERICALLY SHOWN (CE2); SKETCH for the general bound (S3) | CE2: 0.19–0.35 (GELU), 0.22–0.40 (identity), cascade ≤ 6.1e-15 |
| C4 | FlowState's App. B.1 statement "pointwise operations … do not introduce additional temporal discretization error" is false for its multi-layer released models | MEASURED on released weights (FS1–FS3; 16 windows of one series) | layer 0: 5.7e-14 (float64); layers 1–5: 2–49%; forecast 4.2% / 7.0% |
| C5 | Trained per-observation SSMs on HAR change under lossless refinement by 2–6e-3, of which 1.5–3.5e-3 is at the same physical times, and flip ≤ 0.21% | MEASURED (3 seeds; subject-disjoint test) | §6.3 |
| C6 | With identical weights, nonlinear updates on a fixed clock are more robust to missing observations than per-observation updates | MEASURED | §6.2: +0.49 to +0.80 pp lossy mean, CI > 0 |
| C7 | The exact FOH stem improves accuracy over ZOH or no stem | **NOT SUPPORTED** | §6.2: +0.29 [−0.10, +0.73]; −0.64 [−1.23, +0.00] vs no stem |
| C8 | Proposed architecture beats simple resampling (G1) | **REFUTED** on HAR | §7 |
| C9 | Architecture novelty (G2) | **REFUTED** | `RELATED_WORK.md` |
| C10 | Window-sum formulation keeps float32 refinement invariance ≤ 1.6e-5 up to m = 4096; a stable φ2 is necessary | MEASURED (S5; standard summation trick) | §4 |
| C11 | Any recovery of aliased content | NOT CLAIMED (impossible; §0 of THEORY) | Hasegawa 2021: the rate stays detectable after real downsampling |
| C12 | GPU / low-precision kernel behaviour | NOT_RUN (no GPU) | `results/gpu_gate.json` |

## 9. Threats to validity and limits

- **One real classification task.**
  - UCI-HAR's native 50 Hz grid already resolves the learned dynamics well. That is probably why per-observation effects are small there, while FlowState (6 layers, longer memory) and CE2 (coarse grid) show large ones.
  - Tasks with dt coarse relative to the dynamics, very long sequences (EigenWorms) or high-rate audio (Speech Commands, PDM) were NOT_RUN.
- **Small models.**
  - ≈55k-parameter SSMs, 15 epochs, one hyper-parameter setting per family.
  - Absolute HAR accuracy (≈92–93%) is below state-of-the-art HAR models (macro-F1 93–97 in BabyMamba-HAR 2026), and our Transformer baselines are weaker still (88–90%).
  - Conclusions are relative and specific to this setting.
- **Statistical unit.** Test subjects (9), hence wide CIs (±5–6 pp absolute). The paired contrasts are much tighter. Seeds are not data units.
- **Close-work baselines** are our own implementations (NRDE-style, RFormer-style, bilinear multirate-style), not the published code. Their HAR numbers must not be read as the published methods' performance.
- **FlowState probe.**
  - One series (ETTm1/OT), 16 windows, inference only.
  - The comparison under FOH knots is not lossless for FlowState's rule; only the ZOH-repeat comparison is.
  - The probe tests the App. B.1 sentence, not FlowState's stated O(Δ) approximate equivariance.
- **Post-hoc analyses.**
  - The rule-matched twins and the H2 decomposition were added after an adversarial code review (D7).
  - They correct the pre-registered artifact-share metric; they do not change the decision.
- **Main-author verification of literature** is limited to abstracts and key sentences of FlowState v3, Walker 2026, Boulanger & Wood 2026 and Colagrande 2026. The rest was read by sub-agents.
- **Deviations** D1–D10: `docs/DEVIATIONS.md`.

## 10. Sources, licenses, versions

| Item | Source / version | License | Use |
|---|---|---|---|
| This repository | `src/fxclock`, `scripts`, `tests`; CPython 3.11.15, torch 2.14.0+cpu, numpy 2.4.6, scipy 1.17.1, mpmath 1.3.0 | repository license NOT SPECIFIED (owner's decision) | executed |
| `src/n1ref` | N1 repository `donggil113/Sampling-Consistent-Selective-State-Space-Models` @ `306a772` (vendored) | same owner | executed as a reference |
| UCI-HAR | UCI ML Repository #240, zip sha256 `c00b8030…81031` | CC BY 4.0 | executed |
| ETTm1 | github.com/zhouhaoyi/ETDataset `ETT-small/ETTm1.csv`, sha256 `6ce1759b…6c9e` | CC BY-ND 4.0 (LICENSE file of the ETDataset repository, checked 2026-09-26); the data are used unmodified and not redistributed | executed (FS1–FS3) |
| FlowState | HF `ibm-granite/granite-timeseries-flowstate-r1` (main = v1.0, and revision r1.1); code `granite-tsfm` 0.3.9 (pip, `--no-deps`; its torch pin <2.12 was overridden, and it ran with torch 2.14) | Apache-2.0 | executed, inference only |
| Papers | `RELATED_WORK.md` | not checked | read |
