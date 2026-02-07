#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="${ROOT_DIR}/src/pdvi/graph/compute_tangent.py"
SEEDS=(1 2 3 4 5)

for s in "${SEEDS[@]}"; do
  echo "=============================="
  echo "[run] seed=${s}"
  echo "=============================="
  python "${SCRIPT}" --seed "${s}"
done

echo "All runs finished."
