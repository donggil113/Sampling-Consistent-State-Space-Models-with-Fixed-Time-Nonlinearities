# Theory ledger (N2)

Status labels: `STANDARD` (known, cite, never claim), `PROVED` (complete elementary proof given here),
`SKETCH` (argument not checked in full), `MEASURED` (numerical observation only), `CONJECTURE`.
"Novelty" is our reading after the literature check in `RELATED_WORK.md`; absence of evidence is not proof of novelty.

## 0. Setting

- An observation set is O = {(t_i, x_i, m_i)}, with t_i strictly increasing, x_i ∈ R^C, and m_i ∈ {measured, virtual}.
- A reconstruction rule R maps O to a path X = R(O) : [0, T] → R^C.
  - FOH: linear interpolation.
  - ZOH: right hold, X(t) = x_i on (t_{i-1}, t_i].
  - Outside the knots: constant hold (`src/fxclock/knots.py`).
- **Three grid changes** (kept apart everywhere in this project):
  1. *Lossless knot refinement.* O' = O ∪ V, where every knot in V is virtual and placed on R(O). Then R(O') = R(O) as functions. The refinement is lossless only *for the rule used to place V*: an FOH virtual knot is not lossless for ZOH, and vice versa.
  2. *Real downsampling.* O' is a regular subset of the measured knots of O. In general R(O') ≠ R(O).
  3. *Missing observations.* O' is an irregular subset of the measured knots. In general R(O') ≠ R(O).
- **Non-recovery of aliasing (STANDARD; sampling theory).** Let x and x + a be two signals whose difference a vanishes at every kept knot (for example, a component above the new Nyquist rate that is zero on the kept grid). Then they produce identical O', so every model receives identical input. No model, including ours, can recover a. We therefore claim no aliasing recovery, and all downsampling results are reported as information loss, not as artifacts.

## 1. Known results used here (STANDARD — not contributions)

| ID | Statement | Where it is known from |
|---|---|---|
| K1 | Exact discretization of a linear time-invariant system for piecewise-constant (ZOH) or piecewise-linear (FOH / triangle-hold) inputs via variation of constants, h(e) = e^{Λd}h(s) + d[(φ1−φ2)(Λd)BX(s) + φ2(Λd)BX(e)]. The semigroup property implies that splitting a segment leaves h unchanged | digital-control textbooks (Franklin, Powell & Workman, *Digital Control of Dynamic Systems*, FOH/triangle hold); exponential integrators (Hochbruck & Ostermann, Acta Numerica 2010); in deep SSMs: S5 (ZOH), FSSM 2025 (closed-form FOH inside Mamba), FlowState App. B.1 (ZOH exactness), TIDES App. B.1 |
| K2 | Stable evaluation of φ1 and φ2 (series near 0; the naive φ2 loses ≈ ε/\|z\|² relative accuracy) | Hochbruck & Ostermann 2010; standard numerical practice (expm1) |
| K3 | Variation-of-constants / Grönwall perturbation bound for a stable linear system (Re λ ≤ −μ): ‖h_X(t) − h_Y(t)‖ ≤ ‖B‖ ∫_0^t e^{−μ(t−s)} ‖X−Y‖(s) ds ≤ (‖B‖/μ) ‖X−Y‖_∞ | textbook ODE theory; FlowState Prop. B.3 is an instance for two ZOH reconstructions |
| K4 | Piecewise-linear interpolation error ≤ h²‖x''‖_∞/8; ZOH hold error ≤ h‖x'‖_∞ | textbook numerical analysis |
| K5 | Decimation without a prefilter aliases, and aliased content cannot be recovered | Shannon–Nyquist sampling theory |
| K6 | Signatures and log-signatures of a path are invariant under reparametrisation, so inserting collinear knots in a piecewise-linear path does not change them (Chen's identity). Nonlinear updates on fixed windows of log-signatures | Lyons (rough paths); NRDE (Morrill et al. 2021); Log-NCDE (Walker et al. 2024); Rough Transformer (Moreno-Pino et al. 2024, which states only "≈") |
| K7 | Global error orders of one-step methods (Euler O(h), bilinear/trapezoid O(h²)) | Hairer, Nørsett & Wanner |
| K8 | A pointwise nonlinearity applied on a discretization does not commute with the change of discretization (aliasing). Fix: evaluate nonlinearities on a fixed (upsampled) grid | ReNO (Bartolucci et al. 2023), CNO (Raonić et al. 2023), Alias-free GAN (Karras et al. 2021) |
| K9 | Linear continuous-time SSM front end at the input rate, decimated to a fixed rate, with nonlinear layers only after decimation | Multirate PDM-speech SSM (Boulanger & Wood, arXiv 2608.28472): bilinear, no exactness claim |

## 2. Statements specific to the proposed structure

### S1 — Factorisation implies exact refinement invariance (PROVED; low novelty)

**Statement.** Let the model have the form F_θ(O) = Ψ_θ(z_1, …, z_J), where:

- z_j = [Re h(τ_j), Im h(τ_j), X(τ_j)];
- h solves h' = Λh + BX, h(0) = 0, with X = R(O);
- τ_j = jΔ_c is a clock that does not depend on O;
- Ψ_θ is arbitrary (nonlinear, deep).

Then F_θ(O) depends on O only through R(O) restricted to [0, τ_J]. Hence F_θ(O ∪ V) = F_θ(O), and ∇_θ F_θ(O ∪ V) = ∇_θ F_θ(O), for every lossless refinement V under R.

**Proof.** h and X(τ_j) are functions of the path R(O) (K1 gives the closed form, and the closed form is exact for any segmentation of each affine piece). Ψ_θ only sees z. The gradient statement follows because the two sides are the same function of θ. ∎

**Caveats.**

- The same argument applies to *any* path functional on a fixed clock. That includes plain fixed-grid resampling (z_j = X(τ_j)), which is the "trivial exact stem", and fixed-window log-signatures (K6). **Exact refinement invariance is therefore not specific to the exact SSM stem.** It is shared by the mandatory baseline, fixed-grid resampling + SSM.
- In floating point the invariance holds only up to rounding; the floors are measured in S5.

### S2 — Per-knot nonlinearity with FOH re-reconstruction is refinement-invariant only if affine (STANDARD; stated for completeness)

**Statement.** Consider a layer that:

1. applies a pointwise map σ at every knot to an intermediate signal whose knot values are exact samples of a path that is affine between knots (for example the input itself);
2. rebuilds a path with R_FOH;
3. feeds that path to any exact linear stem whose impulse responses {e^{λ(τ−s)}} separate piecewise-linear functions.

Suppose this layer is invariant under insertion of one virtual knot, for all inputs in the range U. Then σ is affine on U.

**Proof.**

- Take knots (0, a) and (1, b), and insert a virtual knot at α ∈ (0,1). Its value is (1−α)a + αb.
- The two re-reconstructed paths are piecewise linear. They agree at 0 and 1. At α they take the values (1−α)σ(a) + ασ(b) and σ((1−α)a + αb).
- Invariance for every λ forces the two paths to be equal, so σ((1−α)a + αb) = (1−α)σ(a) + ασ(b) for all a, b ∈ U and all α.
- A function that satisfies this identity is affine on U. ∎

**Worked example (CE1, exact fractions).**

- Setup: x(t) = t, σ(u) = u², then ∫R_FOH.
- Knots {0, 1} give 1/2; knots {0, ½, 1} give 3/8. The continuous-time limit is 1/3.
- With right-hold ZOH re-reconstruction: 1 vs 5/8.
- With σ evaluated on a fixed clock {0, ½, 1}: 3/8 for both knot sets.

**ZOH version (NUMERICALLY SHOWN, CE2; float64).**

- *Layer 1 is invariant.* Under ZOH-lossless splitting (held values repeated), a per-knot σ applied to the *held input* is invariant, because the values do not change. CE2 confirms this: layer 1 changes by ≤ 6.8e-15 (maximum over 8 units and m = 2…16).
- *Deeper layers are not.* Any deeper layer receives a signal that varies inside held intervals (the output of a nontrivial linear layer). Re-holding that signal at the knots changes it under refinement, **even for σ = identity**: the change is 19–35% with GELU and 22–40% with identity (CE2, median over 8 units, m = 2…16).
- *Diagnosis.* The immediate source is per-knot *re-sampling of intermediate continuous signals*. For σ = identity it can be removed by integrating the whole linear cascade exactly: CE2 cascade_exact ≤ 6.1e-15. For nonlinear σ no closed form is known to us, which is what forces the nonlinearity to be evaluated somewhere. Evaluating it on the observation knots makes the representation grid-dependent; evaluating it on a fixed clock does not (CE2 clock_gelu ≤ 8.0e-15).

**Relation to prior work.**

- The principle "nonlinearity and discretization do not commute" is K8 (ReNO/CNO) in the neural-operator setting.
- The identity σ((1−α)a+αb) = (1−α)σ(a)+ασ(b) is the definition of an affine map (Jensen's functional equation). S2 is therefore a textbook fact and an instance of K8, not a contribution.
- FlowState App. B.1 notes that per-sample pointwise operations "do not introduce additional temporal discretization error" within a layer. Its multi-layer remark acknowledges a "fresh ZOH discretization error introduced at each subsequent SSM layer", which is the CE2 mechanism. The FlowState probe (`RESEARCH_PACKET.md` §5) measures the size of that error on the released weights under ZOH-lossless refinement; it does not refute the appendix.

### S3 — Per-observation stacks converge to their own continuous-time limit under refinement (SKETCH)

- With exact layers and physical dt, re-holding each intermediate signal introduces an O(δ) error per layer, where δ is the mesh (K3 + K4 per layer, then compose Lipschitz constants).
- The model evaluated on the native grid therefore differs from its own δ → 0 limit by O(δ_native). Refinement moves the output toward that limit, and it does not return.
- This is consistent with CE2: the change versus m = 1 grows with m toward a plateau (0.19 → 0.35).
- FlowState App. B.1 sketches the same layer recursion as an upper bound. We do not prove a lower bound. Status: SKETCH.

### S4 — Stability of the clocked model under lossy grid changes (COMPOSITION OF STANDARD RESULTS)

- ‖z_j(O1) − z_j(O2)‖ ≤ ‖C_feat‖ [‖B‖ ∫_0^{τ_j} e^{−μ(τ_j−s)} ‖R(O1) − R(O2)‖(s) ds + ‖R(O1)(τ_j) − R(O2)(τ_j)‖] by K3.
- Then |F(O1) − F(O2)| ≤ L_Ψ max_j ‖z_j(O1) − z_j(O2)‖.
- For a C² signal and a downsampled or missing-data knot set with local gap h′, K4 gives ‖R(O') − x‖ ≤ h′²‖x''‖/8 locally.
- The only structure-specific features are:
  - the bound involves the *path* difference only, never the knot count;
  - the stem term is an exponentially weighted L1 norm, so an isolated missing knot contributes in proportion to (gap width) × (local interpolation error). Fixed-grid resampling (point values only) is controlled by the pointwise error at τ_j instead.
- For content above the new Nyquist rate the bound is O(amplitude), i.e. no recovery (K5).

### S5 — Numerical floor of the window-sum formulation (MEASURED; standard trick)

- **Formulation.** Each segment's contribution is propagated directly to its window end by e^{Λ(τ_j − e)}, with argument in [0, Δ_c]. So the number of sequential compositions is J, independent of the number of knots. The within-window sum has O(n_w ε) worst-case rounding error.
- **Measured** (`results/numerics.json`, 8 real HAR windows, 32 modes):

  | Formulation (float32) | m = 1 | m = 1024 | m = 4096 |
  |---|---|---|---|
  | window-sum | 2.5e-7 | 1.5e-6 | 8.7e-6 |
  | sequential per-segment composition | — | 6.5e-5 | — |

  Float64 stays ≤ 5.3e-15 (maximum over windows) up to m = 4096.
- **Condition.** A stable φ2 is required. Naive φ2 in float32 has relative error 4.1 at |z| = 1e-4. With the window-sum and naive φ, the float32 floor rises to 1.2e-3 at m = 4096. Refinement drives |z| → 0, so the series branch is a necessary condition for numerical invariance, not an optimisation.

### S6 — Cost structure (MEASURED; structural)

- Clocked model: stem O(n_seg · N_stem · C) plus a backbone that runs on J clock steps, independent of observation density.
- Per-observation model: O(n) steps in *every* layer.
- Measured costs, including preprocessing, are in `RESEARCH_PACKET.md` §cost.

### S7 — Ablation equivalence on the native grid (PROVED; elementary)

- If the clock coincides with the native knot times (Δ_c = native dt and τ_j = t_j bit-for-bit; see `knots.clock_times`), the clocked and per-observation inference rules define the same function on the native grid.
- So the 2×2 ablation "fixed clock on/off" is an inference-time switch on identical trained weights.
- **What the switch moves** (corrected after the code review, `docs/DEVIATIONS.md` D7). Under a grid change it moves three things at once:
  1. the stem query times;
  2. the sequence on which the nonlinear backbone runs (with the per-knot re-hold of every linear layer's input);
  3. the readout quadrature nodes. The per-observation readout is a time-weighted right-Riemann mean over knots, which is itself grid-dependent even for a grid-free signal (cf. N1 C5).
- `scripts/supplementary_eval.py` separates (3) from (1)+(2): it reads the refined run only at the measured (native) knots with the native weights (`same_times`), and it reports a grid-free readout control.

### S8 — The exact stem is not an anti-aliasing guarantee (STANDARD)

- The stem is a bank of first-order complex filters, with |H_n(ω)| ≈ |b_n| / |iω − λ_n|.
- Sampling h at the clock rate aliases the part of h above π/Δ_c. First-order roll-off is weak, so the exact stem reduces point-sampling aliasing but does not remove it.

## 3. What is and is not claimed

- **Claimed**
  - S1 with an explicit (elementary) proof; S5–S7 as measurements. S2 is standard and is stated only for completeness.
  - The empirical answer to the core question on a trained real-signal model (HAR) and on a released foundation model (FlowState), with lossless refinement defined relative to each model's own reconstruction rule.
- **Not claimed**
  - Exact FOH/ZOH integration (K1), Grönwall bounds (K3), interpolation error (K4), log-signature invariance (K6), the "linear front end → decimation → nonlinear layers" structure (K9), or any aliasing recovery.
  - Any architecture novelty. That depends on the decision rule in `configs/prereg_n2.json` and the results in `RESEARCH_PACKET.md`.
