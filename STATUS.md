# STATUS — N2: Beyond Step-Size Rescaling (fixed-time nonlinearities)

- Last updated: 2026-09-26 (UTC).
- Branch: `claude/busy-galileo-3hk0tn`.
- Details: `RESEARCH_PACKET.md`.

## Verdict: **NO_GO** (for any new-architecture claim)

The pre-registered decision rule (`configs/prereg_n2.json`) fails on both criteria.

- **G1 (beats simple resampling): FAIL.**
  - On UCI-HAR the proposed model does not beat fixed-grid resampling + the same SSM.
  - P1 (exact FOH stem, native-rate clock) vs point resampling: −0.64 pp on the lossy-condition mean, subject-bootstrap 95% CI [−1.23, +0.00]; −0.41 pp native.
  - P4 (coarse clock) vs point / box-filter resampling at the same clock: −0.35 / −0.68 pp.
  - The one robust gain of the fixed clock is robustness to missing observations, for weights trained on the native grid only. Running the same trained weights with nonlinear updates on the clock instead of per observation gains +0.80 pp [+0.42, +1.20] on the lossy mean, and +2.50 pp at 70% drop.
  - Fixed-grid resampling without any stem shows the same gain: B_point_clock1 vs B_dtonly, same weights, +0.49 pp [+0.13, +0.86]. **The gain is explained by simple resampling.**
  - Part of it may be a train/test grid mismatch of the zero-shot per-observation rule. Grid-augmented training (AUG arm) was NOT_RUN.
- **G2 (not prior work): FAIL.** After adversarial re-checking, 18 distinct sources (27 verification records) have overall overlap "substantial" with "exact input summary + nonlinear updates on an observation-independent clock". Six have "full" overlap on the fixed-time-nonlinearity component. The sources include:
  - Walker et al. 2026, with an explicitly stated exact invariance to value-preserving insertions;
  - Rough Transformer 2024;
  - Logsig-RNN 2019;
  - NRDE / Log-NCDE;
  - the multirate PDM-speech SSM 2026;
  - sampling-frequency-independent audio layers 2021–2025;
  - SNO / ReNO / CNO / CROP in operator learning.

  See `RELATED_WORK.md`.

## Answer to the core question (reported separately from GO/NO_GO)

> Does a nonlinear update at every observation make the representation depend on the grid of the same reconstructed input?

**Yes, but the mechanism depends on the reconstruction rule, and the size depends on the setting.**

- **FOH rule.** A per-knot σ alone breaks invariance unless σ is affine. This is STANDARD, an instance of the operator-learning principle.
- **ZOH rule.** A per-knot σ on the held input is harmless. What breaks invariance is per-observation re-sampling of each layer's output between layers, and it does so for linear and nonlinear layers alike.
- **The nonlinearity** is what makes this unavoidable, because a nonlinear stack cannot be integrated exactly as one cascade.

| Setting | Grid change (lossless for the model's own rule) | Effect |
|---|---|---|
| Minimal counterexample (CE1, exact) | one virtual knot | output 1/2 → 3/8. It is 3/8 for both when σ is on a fixed clock. For FOH re-reconstruction, invariance holds ⇔ σ is affine (`docs/THEORY.md` S2) |
| 2-layer S5/FlowState-style stack (CE2, float64) | ZOH splitting m = 2…16 | layer 1 exact (≤ 6.8e-15); output change 19–35%. With σ = identity: 22–40% (the cause is per-knot re-sampling of intermediate signals) |
| **Released FlowState** (ETTm1, 16 windows) | every sample repeated ×2 (added post hoc; the pre-registered FOH test gives 4.7–4.8% / 6.7%, not consistent at ≤ 1e-3) | forecast change: median 4.2% (v1.0) and 7.0% (r1.1), max 24%. The median is larger than under real 2× downsampling (2.2% / 3.9%). With RevIN statistics fixed (FS3, 4 windows), layer 0 is exact in float64 (5.7e-14) and layers 1–5 change by 2–49%. This quantifies the inter-layer re-hold error that FlowState App. B.1's multi-layer remark acknowledges and bounds at O(Δ) |
| Trained HAR per-observation models (3 seeds) | pre-registered rule-matched virtual knots (A1: FOH; A3, B_dtonly: ZOH) | median relative logit change 2.1–4.9e-3 in float32 (float64 medians 2.6–5.6e-3). About 40–70% of it is a change at the same physical times (1.5–3.5e-3). For A1 the rest is of the size of a grid-free readout-quadrature control. For A3 and B_dtonly that control is 0, so the rest is the representation at the virtual knots. Flip rate ≤ 0.10%: **numerically real, practically negligible** on HAR |

## What was done

| Step | Status | Where |
|---|---|---|
| 1. FlowState + latest multirate SSM literature, overlap check | DONE | `RELATED_WORK.md`: 103 sources, 76 adversarial re-checks, completeness critic; key quotes self-verified |
| 2. Separate lossless refinement / real downsampling / missing observations; virtual vs measured knots; no aliasing-recovery claim | DONE | `src/fxclock/knots.py`, `docs/THEORY.md` §0 |
| 3. Minimal counterexample | DONE | `scripts/counterexample.py`, `results/counterexample.json` |
| 4. Minimal model; input reconstruction and output query rules published | DONE | `src/fxclock/`, `RESEARCH_PACKET.md` §2, §4 |
| 5. float64 reference, interval-split consistency, small dt, forward/backward | DONE | 23 unittest checks; `results/numerics.json`. GPU gate: NOT_RUN (no GPU; `results/gpu_gate.json`) |
| 6. Standard results vs structure-specific ones | DONE | `docs/THEORY.md` (K1–K9 standard; S1–S8 with status) |
| 7. Baselines: fixed-grid resampling + SSM (mandatory), dt-only, resampling + Transformer, close work (NRDE-style, RFormer-style, multirate-bilinear-style) | DONE | `RESEARCH_PACKET.md` §6 |
| 8. Real-signal 2×2 ablation (exact stem × fixed clock) with cost including preprocessing | DONE | `RESEARCH_PACKET.md` §6, `results/har_summary.md`, `results/cost_har.json` |

## Deviations (full log in `docs/DEVIATIONS.md`)

- **D1.** RFormer-style baseline: basepoint fix, seed 0 re-run.
- **D2, D6.** Slow per-observation evaluation replaced by numerically equivalent code. Affected runs re-evaluated eval-only from checkpoints.
- **D3.** Custom scan adjoint.
- **D4.** FlowState refinement sequences corrected (pre-window hold).
- **D7.** After an adversarial code review: rule-matched ZOH twins for the artifact share, and an H2 decomposition (post hoc).
- **D8.** Readout span normalisation.
- **D11.** FlowState ZOH-repeat condition and FS2/FS3 added post hoc.
- **D12.** Up to 5–6 concurrent processes instead of 4.
- **D9.** Transformer baselines are non-causal.
- **D10.** Secondary AUG arm NOT_RUN (cost).

The main arm finished 4 h 42 min after launch, within the 5-hour budget. It used up to 5–6 concurrent processes instead of the registered 4 (D12). The float64 H2 medians (supplementary) were computed after the 5-hour mark.

## NOT_RUN / limits

- **GPU / float16 / bfloat16 kernels.** NOT_RUN: no GPU in this container. The float32 CPU floors are measured.
- **AUG arm** (grid-augmented training of per-observation models). NOT_RUN (cost: one epoch did not finish in 27 min).
- **SFI-STFT frame-clock baseline** (suggested by the literature critic). NOT_RUN. Whether an SFI-STFT front end is refinement-invariant is untested here. The distinction from USES/USEMamba in `RELATED_WORK.md` is an unverified agent claim.
- **One real classification task only** (UCI-HAR, 50 Hz, subject split), plus one forecasting probe (FlowState on ETTm1).
  - EigenWorms, Speech Commands and PDM audio were NOT_RUN.
  - HAR models are small (≈55k parameters, 15 epochs). Absolute accuracy (≈92–93%) is below state-of-the-art HAR models (macro-F1 93–97 reported by BabyMamba-HAR 2026), so only relative comparisons are meaningful.
- **Not independent units.** HAR test windows overlap (50%); the unit is the subject (9), hence wide CIs. FlowState windows come from one series.
- **Literature.** Most per-paper details were read by sub-agents (`agent`). The main author re-checked abstracts and key sentences only for FlowState v3, Walker 2026, Boulanger & Wood 2026 and Colagrande 2026.

## Next (only if the owner wants an analysis note, not an architecture)

1. Write the analysis as an instance of the known operator-learning principle (SNO/ReNO/CNO) for time-series SSMs.
   - Content: the measured size of the inter-layer re-hold error in the released FlowState (with credit to its App. B.1 multi-layer remark), the three-way grid-change protocol with knot provenance, and the HAR measurements.
   - CE1/S2 is STANDARD; use it only as an illustration.
   - It must credit Walker 2026, RFormer, Logsig-RNN, torchcde, the SFI layers and the multirate PDM SSM.
2. If a GPU becomes available, run `scripts/gpu_gate.py`.
3. Optionally, test FlowState on more series and horizons, and run the AUG arm with a GPU.
