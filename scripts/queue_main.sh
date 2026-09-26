#!/usr/bin/env bash
# Main arm: 13 groups x 3 seeds, 4 parallel single-thread processes, seed-major, slowest groups first.
cd "$(dirname "$0")/.."
mkdir -p results/raw/har
GROUP_LIST="G_tf1 G_foh1 G_zoh1 G_none1 G_tfpatch4 G_rformer4 G_P4 G_zoh4 G_bilin4 G_pt4 G_bin4 G_patch4 G_nrde4"
for s in 0 1 2; do for g in $GROUP_LIST; do echo "$g $s"; done; done | \
  xargs -P 4 -L 1 bash -c 'python3 scripts/run_har.py --group $0 --seed $1 --config configs/prereg_n2.json --out results/raw/har > results/raw/har/$0__seed$1.stdout 2>&1'
echo "ALL DONE $(date -u)" > results/raw/har/QUEUE_DONE
