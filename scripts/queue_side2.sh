#!/usr/bin/env bash
# Side queue 2 (1 process): eval-only re-runs (deviation D6) for runs whose per-observation evaluation was stopped.
cd "$(dirname "$0")/.."
for job in "G_foh1 0" "G_foh1 1" "G_zoh1 1" "G_none1 1" "G_foh1 2"; do
  set -- $job
  python3 scripts/run_har.py --group $1 --seed $2 --config configs/prereg_n2.json --out results/raw/har --eval-only > results/raw/har/${1}__seed${2}.eval.stdout 2>&1
done
echo "SIDE2 DONE $(date -u)" > results/raw/har/SIDE2_DONE
