#!/usr/bin/env bash
# Secondary AUG arm (descriptive): 4 rules trained on random grids, seeds 0-1, 4 parallel processes.
cd "$(dirname "$0")/.."
for s in 0 1; do for g in AUG_dtonly AUG_A3 AUG_P1 AUG_pt1; do echo "$g $s"; done; done | \
  xargs -P 4 -L 1 bash -c 'python3 scripts/run_har.py --group $0 --seed $1 --config configs/prereg_n2.json --out results/raw/har_aug > results/raw/har_aug_$0__seed$1.stdout 2>&1'
echo "AUG DONE $(date -u)" > results/raw/AUG_DONE
