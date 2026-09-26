#!/usr/bin/env bash
# Side queue (1 process): eval-only for seed-0 groups whose Hillis-Steele evaluation was stopped, and the
# seed-0 re-run of B_rformer_clock4 after deviation D1 (basepoint augmentation).
cd "$(dirname "$0")/.."
for g in G_foh1 G_zoh1 G_none1; do
  python3 scripts/run_har.py --group $g --seed 0 --config configs/prereg_n2.json --out results/raw/har --eval-only > results/raw/har/$g__seed0.eval.stdout 2>&1
done
python3 scripts/run_har.py --group G_rformer4 --seed 0 --config configs/prereg_n2.json --out results/raw/har > results/raw/har/G_rformer4__seed0.stdout 2>&1
echo "SIDE DONE $(date -u)" > results/raw/har/SIDE_DONE
