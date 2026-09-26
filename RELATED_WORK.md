# Related work and overlap check (N2)

Checked 2026-09-26. There were six parallel literature readers, one per angle:

1. FlowState and its code;
2. multirate SSMs;
3. CDE / log-signature models;
4. alias-free and discretization-invariant operators;
5. irregular time-series models;
6. exact FOH discretization.

Then came an adversarial re-check of every partial-or-higher overlap and a completeness critic.

- **Adversarial re-check.** 76 verification agents re-opened the primary source and were told to argue the hostile-reviewer direction when unsure. Of the ratings they changed, 9 went up and none went down. Ratings in this file are the corrected ones.
- **Completeness critic.** 1 agent, which added 8 sources (§6b).
- **Total.** 103 distinct sources. Raw records (readers, verifications, critic) are in `results/literature_records.json`.

- **Access**
  - `FULL_TEXT_SECTIONS`: the listed sections of the full text were read, located by keyword search.
  - `ABSTRACT_ONLY`: only the abstract was read.
  - `NOT_ACCESSED`: the source could not be read.
- **Provenance**
  - `self`: re-checked directly by the main author in this session (abstract fetched, and key sentences searched in the arXiv HTML full text).
  - `agent`: read by a literature sub-agent only. Anything tagged only `agent` must be re-verified before it is cited in a manuscript.

## 1. Bottom line

The central *architecture* is already covered by prior work in substance:

- an exact (or exactly-invariant) linear summary of the reconstructed input;
- nonlinear updates only on a grid that does not depend on the observations;
- hence exact invariance to value-preserving / collinear knot insertion.

It appears as log-signature windows + RNN/ODE (Logsig-RNN 2019, NRDE 2021, the torchcde `logsig_windows` pipeline), as signature patching + attention (Rough Transformer 2024), as a linear SSM front end + decimation (multirate PDM SSM 2026), and in the most complete form in Walker et al. 2026. Walker et al. 2026 has:

- exact interval log-signatures;
- a query partition independent of the observation times;
- an *explicitly stated* exact invariance to inserting value-preserving observations;
- count coordinates that separate repeats from real measurements.

The core *observation* is also known in the operator-learning literature (SNO 2022, ReNO 2023, CNO 2023, CROP 2025, Colagrande et al. 2026): pointwise nonlinearities evaluated on the input grid make the representation discretization-dependent under lossless refinement, and the fix is to put the nonlinearity on a fixed grid.

What we did **not** find (absence of evidence, not proof of novelty):

1. An exact FOH (piecewise-linear) *linear-SSM* stem feeding a standard deep SSM on a fixed physical clock, with the exactness stated as a theorem.
   - Nearest: the Exponentially Weighted Signature (Bloch et al. 2026) integrates the path increment dX exactly per segment. The FOH stem integrates X itself.
2. A controlled test on a released sampling-rate-invariant SSM (FlowState) under refinement that is lossless *for its own reconstruction rule*, with a layer-wise float64 attribution.
   - FlowState App. B.1 states that its per-sample pointwise operations "do not introduce additional temporal discretization error". FS3 shows layer 0 exactly invariant (5.7e-14) and layers ≥ 1 not (2%–49%).
3. A single controlled comparison that separates lossless refinement (virtual knots), real downsampling and missing observations for SSM classifiers with a knot-provenance flag.
   - Partial precedents:
     - FalsePromise 2025 separates "resolution interpolation" from "information extrapolation" for operators.
     - Walker 2026 count coordinates.
     - Logsig-RNN tests downsampling and missing data separately.

## 2. Closest prior work (overall overlap substantial)

| Work | What it already does | What N2 adds or differs in | Access / provenance |
|---|---|---|---|
| **Walker, Bloch, Yang, Morley, Lyons 2026**, "Faithful Embeddings of Irregular and Asynchronous Data for Online Log-NCDEs", arXiv 2605.30213 | Observations as Lie-algebra increments composed into exact interval log-signatures over arbitrary query intervals. Log-NCDE on a query partition drawn independently of the observation times. App. A.5: G_{α,β}(x) = G_{α,β}(x′) whenever x′ differs by value-preserving insertions. Count coordinates distinguish a repeat from an absent observation | FOH (collinear) virtual knots instead of value-preserving ones. A linear-SSM stem instead of log-signatures. An explicit downsampling/aliasing arm. Nonlinearity placement tested inside SSM stacks | FULL_TEXT_SECTIONS; self (abstract; "value-preserving", "query partition", "count coordinates" found and read in the HTML full text) + agent |
| **Boulanger & Wood 2026**, "Multirate SSMs for End-to-End Processing of PDM Speech", arXiv 2608.28472 | A continuous-time HiPPO-LegT/FouT SSM, bilinear at the input rate (16 kHz–2 MHz), decimated to a fixed rate (up to 65,536×) without anti-aliasing. All nonlinear layers run only on the decimated clock. Claims a "modulation- and sampling-rate-invariant latent representation" (empirical) | Exact FOH integration instead of bilinear on raw samples. An exactness statement. Non-integer and irregular grids. Separation of the grid changes. PDM vs PCM is not the same reconstructed path | FULL_TEXT_SECTIONS; self (abstract) + agent |
| **Moreno-Pino et al. 2024**, Rough Transformer, arXiv 2405.20799 | Linear interpolation of the observations. Exact signatures (global and local) at a fixed set of time points. Attention and MLP only on that fixed-length sequence. Prop. 3.1 gives "≈" robustness to reparametrisation. Tests random drops | An exact (not "≈") statement. A lossless-refinement test. SSM stems. Downsampling arm | agent |
| **Liao et al. 2019 / 2021**, Logsig-RNN, arXiv 1908.08286, 2110.13008 | Log-signatures of the piecewise-linear path over a coarse time partition, with an RNN only on the coarse steps. Frame-rate robustness stated through reparametrisation invariance. Downsampling and missing data tested separately | Same as above | agent |
| **Morrill et al. 2021** NRDE (arXiv 2009.08295); **Walker et al. 2024** Log-NCDE (arXiv 2402.18512); the torchcde `logsig_windows` pipeline | Window log-signatures and fixed-step nonlinear updates per window. With physical-time windows this is exactly invariant to collinear knots (never stated). The published pipelines use index-based windows, which move with inserted knots | N2 uses physical-time windows by construction and states and tests the invariance | agent |
| **Saito et al. 2021**, arXiv 2105.04079; **Imamura et al. 2023**, arXiv 2306.10718 (sampling-frequency-independent layers) | A linear front end generated from latent analog filters for any sampling rate. The nonlinear mask network runs on frames with a hop fixed in seconds. Non-integer strides via sinc interpolation. "Resample to the trained rate" baseline | FOH-exact SSM stem; classification; separation of the grid changes | agent |
| **Fanaskov & Oseledets 2022**, Spectral Neural Operators, arXiv 2205.10573 | States that under lossless refinement a pointwise σ followed by a global linear operator makes the output grid-dependent (FNO differs by about 25% between h and 2h). Fix: a fixed-size coefficient representation | Time series, causal FOH paths, SSMs | agent |
| **Bartolucci et al. 2023** ReNO (arXiv 2305.19913); **Raonić et al. 2023** CNO (arXiv 2302.01178); CROP (ICLR 2025) | A formal framework for aliasing and representation equivalence. Nonlinearity on a fixed (upsampled, band-limited) computational grid. Resampling to the computational grid for other resolutions | Band-limited spaces there, piecewise-linear here. N2's exact invariance is an instance of representation equivalence with hat-function frames on nested knots (agent's reading) | agent (CROP: ABSTRACT_ONLY) |
| **Colagrande et al. 2026**, "Limits of Resolution Equivariance in FNOs", arXiv 2606.00677 | Lossless (Fourier-padded) refinement changes FNO outputs; attributes this to nonlinear aliasing; running at the training grid and upsampling is a strong baseline | Same principle, time-series SSM setting | self (abstract) + agent |
| **Kidger et al. 2020** NCDE (arXiv 2005.08926); Kidger thesis (arXiv 2202.02435); Berner et al. 2025 (arXiv 2506.10973) | Upgraded to substantial by the adversarial re-check. NCDE solutions are continuous-level invariant to spline-consistent knots. Berner et al. document "interpolate to a fixed latent grid, then run the network" as a standard recipe, and note it discards high-resolution information | — | agent |
| **Zhang et al. 2023** USES (arXiv 2309.17384); **Chao et al. 2025** USEMamba (arXiv 2505.21198) | Fixed-duration STFT (hop 16 ms), so the frame count does not depend on the sampling rate. All nonlinear layers (dual-path Transformer or Mamba) run on this physical-time frame clock. Train at 8 kHz, test at 16/48 kHz | An STFT of point samples is not refinement-invariant (kink harmonics of the PL path enter new bins). In this project the analogous "fixed clock without exact stem" rules (A2_zoh_clock, B_bilin_clock4) are measured instead; an SFI-STFT baseline was NOT run | agent (critic) |

## 3. FlowState (the model named in the task)

- **Sources**
  - arXiv 2508.05287 v1–v3 (ICML 2026 per v3): FULL_TEXT_SECTIONS, agent + self (keyword search of the v3 HTML).
  - Code: granite-tsfm 0.3.9 `modeling_flowstate.py`, Apache-2.0. Read and executed by the main author.
- **Mechanism**
  - S5-style complex diagonal SSM layers, exact ZOH.
  - The scale factor multiplies every per-state Δ.
  - Per-sample output gate, MLP (SwiGLU or selu-gated), LayerNorm and residual.
  - Causal RevIN with running *sample-count* mean and std.
  - Functional basis decoder (Legendre) queried at any rate.
- **Claims**
  - App. B.1, verbatim (self-checked in the HTML): pointwise operations "act independently at each time step and therefore do not introduce additional temporal discretization error". Also: "The only approximation relative to the original continuous signal u(t) comes from this ZOH reconstruction."
  - Sec. 4: "approximately sample-rate equivariant, with an approximation error vanishing as max{Δ,Δ′} → 0".
  - Prop. B.3 is a standard ZOH/Grönwall bound (K3 in `docs/THEORY.md`).
- **Tested grid changes**
  - Real downsampling only (ETT at 15–135 min) and ETTh1 → ETTm1-rate forecasting.
  - No virtual-knot, interpolation-upsampling or random-drop tests.
- **Code detail relevant here**
  - The kernel is x_k = Ā x_{k−1} + B̄ u_k with B̄ = (Ā−1)/λ·B, i.e. the sample u_k enters x_k fully.
  - The lossless ZOH refinement for this kernel therefore repeats every sample r times, including the first (deviation D4).
- **N2 must credit FlowState for**
  - continuous-time output querying (FBD);
  - the first-order ZOH mismatch bound.

## 4. Known results that must be cited, not claimed

| Result | Canonical source | Provenance |
|---|---|---|
| FOH / triangle-hold exact discretization; the φ1−φ2, φ2 weights are the exact "exponential trapezoidal" rule | Hochbruck & Ostermann, Acta Numerica 2010, Ex. 2.6; Franklin, Powell & Workman, *Digital Control*, §6.3; MATLAB `c2d` docs | agent |
| Closed-form FOH inside deep SSMs | FSSM (arXiv 2509.08458, 2025); "Beyond ZOH" Vision Mamba (arXiv 2604.20606, 2026). Mamba-3's "exponential-trapezoidal" is a *different*, non-exact rule | agent |
| Exact PL-segment integration of a linear SSM driven by dX, with a stable small-Δt formulation | Exponentially Weighted Signature (Bloch et al., arXiv 2603.19198, 2026) | agent |
| Semigroup property; stable φ-functions | Hochbruck & Ostermann 2010; Kassam & Trefethen 2005 | agent |
| Grönwall / variation-of-constants bounds; interpolation error h²‖x''‖/8 | textbooks (Gronwall 1919; de Boor) | agent (NOT_ACCESSED for Gronwall 1919) |
| Aliasing under decimation; no recovery | Oppenheim & Schafer §4.6 (NOT_ACCESSED; textbook fact) | — |
| Δ rescaling for resolution change; physical Δt per step | S4 §4.3, LSSL, S5 §6.3 (16 → 8 kHz zero-shot: S5 96.52 → 94.53), S4ND, Zubić et al. 2024, TIDES 2026 | agent (S5 numbers) |
| Signature reparametrisation invariance, Chen's identity | Lyons; Chevyrev & Kormilitzin 2016 | agent |

## 5. Irregular time-series baselines (task-appropriate close work)

- **TIDES 2026** (arXiv 2605.09742).
  - Physical Δ; input dependence moved to Re Λ, B and C; GELU/GLU per token.
  - It is still a per-observation nonlinear model, so it falls in N2's "dt-only" class.
  - EigenWorms 85.7 ± 4.0.
- **mTAN 2021, IP-Nets 2019.** A learned resampler onto a fixed reference grid, then a GRU. The density-normalised smoothers change under virtual knots (agent's analysis).
- **NCDE 2020.** Nonlinear vector field evaluated continuously inside the solver. The continuous-level solution is invariant to spline-consistent virtual knots. Solver steps with jump points are knot-dependent.
- **GRU-D, ODE-RNN, CRU, Neural Flows, ContiFormer, LinOSS, S7, Liquid-S4.** Per-observation updates or index-based steps; no refinement invariance.

**Calibration numbers.**

- UCI-HAR: BabyMamba-HAR 2026 (arXiv 2602.09872) reports *macro-F1* 93–97 for competent models on raw windows with the authors' subject split. The N2 models are deliberately small (≈55k parameters, 15 epochs) and are used only for *relative* comparisons.
- EigenWorms: Log-NCDE 85.6, NRDE 83.9, RFormer 90.3, LRU 87.8.

## 6. Other checked sources (overlap none or partial)

- **Multirate / multiscale SSMs.** S-Edge (Bittner et al. 2025, layer-wise decimation, RZOH augmentation), HiSS (Bhirangi et al. 2024, resample to 50 Hz then chunked SSMs), SaShiMi, MS-SSM (2512.23824), ms-Mamba, ScanResample (2410.08681), VISTA-SSM, EventSSM (2404.18508), PASS, PDM-KWS (Yarga & Wood 2024).
- **Operator learning.** Alias-free GAN, MambaNO, FalsePromise (2510.06646), Berner et al. 2025, FNO discretization error, FNO, Neural Operator (Kovachki et al.), ResInvFNO, SS-NO (2605.18905).
- **Continuous kernels.** CKConv, FlexConv, CCNN, S4ND.
- **CDE / signature methods.** Online NCDE, Kidger thesis, SLiCE, Cirone et al. 2024 (selective SSMs as linear CDEs only when Δ ∝ dt), neural signature kernels, generalised signature method, Bleistein et al.
- **SSMs.** Mamba, Mamba-2, Mamba-3, LRU, S4D, LS4, HiPPO, S2P2, eSSM.
- **Audio.** Sample-rate-independent RNNs for audio effects (2406.06293, 2409.15884).
- **Forecasting.** TiRex (the source of FlowState's masking/RevIN).

## 6b. Added by the completeness critic

- **HALO** (arXiv 2608.27233, 2026). IMU foundation model with 1-s physical-time patches and a Transformer on the patch clock. The within-patch CNN is nonlinear at every sample (kernels defined in samples), so it is not refinement-invariant. UCI-HAR is only in its training set.
- **Hasegawa et al. 2021** (arXiv 2101.00812). HAR robust to sampling rates: linear interpolation to a fixed length, a CNN, and an adversarial sampling-rate discriminator. It is the HAR instance of the mandatory resampling baseline. The discriminator still identifies the rate at 96.9% after real low-pass downsampling, so resampling is not invariant to real downsampling, which is consistent with this project's three-way separation. HASC accuracy is 85–87% down to 12.5 Hz, collapsing to 39–46% at 4 Hz.
- **t-PatchGNN** (ICML 2024) and **Phased LSTM** (2016). Fixed time-span patches, and time-gated per-observation updates, respectively.
- **ADAA-RNN** (DAFx 2025) and **Pupu-Vocoder** (arXiv 2512.20211). Antiderivative anti-aliasing evaluates each activation in closed form over the linear-interpolation reconstruction of its argument. "Continuous-time treatment of the PL reconstruction" is therefore also known on the *nonlinearity* side, and this project claims exactness only for the linear stem integral.
- **Found but not read** (no claims made): TF-Refiner 2609.29463, SFI codec 2607.01865, NCDSSM 2301.11308, Khatri-Rao Neural Operators, CaMBRAIN 2605.28792, WaveSSM 2602.22266, 2505.11375, ReIMTS 2602.21498, MambaRate 2507.12090, GP adapters (Li & Marlin 2016; Futoma 2017).

## 7. Consequence for N2

- **The architecture claim fails criterion G2** of the pre-registration. After adversarial re-checking, more than twenty sources have overall overlap "substantial", and five have "full" overlap on the fixed-time-nonlinearity component: RFormer, torchcde `logsig_windows`, Walker 2026, SFI non-integer strides, and Berner 2025. The verdict is **NO_GO for any new-architecture claim**, whatever the HAR numbers.
- **What remains** could only be an *analysis* contribution:
  - the per-observation-nonlinearity counterexample and the affine-only proposition (S2), framed as an instance of the known operator-learning principle;
  - the FlowState refutation (FS1–FS3);
  - the controlled separation of the three grid changes.

  Each must credit SNO, ReNO, CNO, Walker 2026, RFormer, Logsig-RNN and the multirate PDM SSM.
