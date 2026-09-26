# N2 — Beyond Step-Size Rescaling: Sampling-Consistent SSMs with Fixed-Time Nonlinearities

**Core question.** When only the time grid of the *same reconstructed input* changes, does performing a nonlinear update at every observation make the representation depend on the grid?

Start here:

- `STATUS.md` — verdict, deviations, what was not run.
- `RESEARCH_PACKET.md` — claims ledger, results, the GO/NO_GO decision.
- `RELATED_WORK.md` — overlap check (FlowState, multirate SSMs, log-signature models, alias-free operators).
- `docs/THEORY.md` — which results are standard and which are specific to this structure.
- `configs/prereg_n2.json` — pre-registration, committed before any test-set evaluation.

## Layout

| Path | Content |
|---|---|
| `src/fxclock/knots.py` | knots with a `measured` flag; published FOH/ZOH reconstruction rules; virtual-knot refinement, downsampling, dropping; the fixed physical clock |
| `src/fxclock/phi.py` | stable φ1/φ2 (series branch near 0, autograd-safe) |
| `src/fxclock/stem.py` | exact FOH/ZOH linear-SSM stem (window-sum formulation); bilinear stem (multirate-SSM-style baseline) |
| `src/fxclock/ssm.py` | S4D-style layers in physical time: fixed clock (LTI, FFT) or per observation (dt_k, scan); Transformer backbone |
| `src/fxclock/models.py` | proposed model, 2×2 ablation, baselines |
| `src/fxclock/prep.py` | preprocessing from raw knots to model inputs (part of the measured cost); output query rules |
| `src/n1ref/` | vendored N1 reference code (pure Python, float64), used as an independent reference |
| `tests/test_numerics.py` | 22 numerical checks (unittest) |
| `scripts/counterexample.py` | minimal counterexamples CE1–CE3 |
| `scripts/run_numerics.py`, `scripts/gpu_gate.py` | φ accuracy, float32 floors, small-dt checks; GPU gate |
| `scripts/run_har.py`, `scripts/queue_main.sh` | real-signal experiment (UCI-HAR) |
| `scripts/analyze_har.py`, `scripts/cost_har.py` | aggregation (subject bootstrap) and cost including preprocessing |
| `scripts/flowstate_*.py` | refinement probes of the released FlowState model on ETTm1 |

## Reproduce

```bash
pip install numpy scipy mpmath torch --index-url https://download.pytorch.org/whl/cpu   # torch 2.14 CPU used here
python3 -m unittest tests.test_numerics                 # numerical checks
python3 scripts/counterexample.py                       # CE1-CE3
python3 scripts/run_numerics.py; python3 scripts/gpu_gate.py
scripts/queue_main.sh                                   # HAR main arm (about 3-4 h on 4 CPU cores)
python3 scripts/cost_har.py; python3 scripts/analyze_har.py
pip install --no-deps granite-tsfm==0.3.9 && pip install "transformers>=4.57.6" pandas scikit-learn deprecated einops datasets
python3 scripts/flowstate_probe.py; python3 scripts/flowstate_attribution.py; python3 scripts/flowstate_fp64_layers.py
```

Data:

- UCI-HAR, CC BY 4.0: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
- ETTm1: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv

The paths used are in `configs/prereg_n2.json`.
