"""GPU gate: float32 CUDA kernels vs the float64 CPU reference for refinement invariance.

No GPU gate existed in the accessible prior repositories (N1 recorded "gpu": false only), so this
gate is new.  If no CUDA device is present it records NOT_RUN and exits 0 without claiming anything.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = "results/gpu_gate.json"


def main():
    rec = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        rec.update({"status": "NOT_RUN", "reason": "no CUDA device in this container (torch.cuda.is_available() is False)"})
        json.dump(rec, open(OUT, "w"), indent=1)
        print(json.dumps(rec))
        return
    import numpy as np
    from fxclock import knots as K
    from fxclock.stem import ExactStem
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    t = np.arange(128) * 0.02
    ks = K.from_uniform(np.sin(2 * np.pi * 7.0 * t)[:, None] + 0.1 * rng.standard_normal((128, 1)), 0.02)
    q = K.clock_times(31, 4, 0.02)
    stem64 = ExactStem(1, 32, 0.02, 2.56, 25.0, seed=0).double()
    stem32 = ExactStem(1, 32, 0.02, 2.56, 25.0, seed=0).to(dev)

    def run(stem, kk, dtype, device):
        seg = K.segments(kk, q, "foh", 0.0)
        T = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=device)[None]
        with torch.no_grad():
            return stem(T(seg["d"]), T(seg["xs"]), T(seg["xe"]), T(seg["win"], torch.long), T(seg["rem"]), T(seg["qdt"]))[0].cpu().to(torch.complex128)

    ref = run(stem64, ks, torch.float64, "cpu")
    rows = []
    for m in (1, 4, 64, 1024):
        km = K.refine_uniform(ks, m, "foh")
        h = run(stem32, km, torch.float32, dev)
        rows.append({"m": m, "cuda_fp32_rel_err_vs_cpu_fp64": float((h - ref).abs().max() / ref.abs().max())})
    rec.update({"status": "RUN", "device": torch.cuda.get_device_name(0), "rows": rows,
                "gate": "PASS" if all(r["cuda_fp32_rel_err_vs_cpu_fp64"] < 1e-4 for r in rows) else "FAIL"})
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
