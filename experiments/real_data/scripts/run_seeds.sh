#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="${ROOT_DIR}/src/pdvi/models/train_graph.py"
CONFIG="${1:-${ROOT_DIR}/experiments/real_data/configs/train_graph.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mapfile -t SEEDS < <(python - "${CONFIG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
seeds = cfg.get("seeds")
if not seeds:
    seeds = [cfg.get("seed", 1)]
for s in seeds:
    print(int(s))
PY
)

for seed in "${SEEDS[@]}"; do
    echo "======================================"
    echo "Running seed = ${seed} on cuda:${CUDA_VISIBLE_DEVICES}"
    echo "======================================"

    python "${SCRIPT}" --config "${CONFIG}" --seed "${seed}"
done
