#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Sim2Real2Sim
for mode in firm moderate original; do
  for dt in .002 .001; do
    "$HOME/.venvs/so101-m1/bin/python" scripts/controller_audit.py --contact "$mode" --impratio 1 --iterations 100 --ls-iterations 50 --timestep "$dt" --tag "ratio1_${mode}_${dt}" > "artifacts/ratio1_${mode}_${dt}.log"
  done
done
