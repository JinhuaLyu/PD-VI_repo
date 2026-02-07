#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SEED="${1:-1}"
CONFIG="${ROOT_DIR}/experiments/real_data/configs/train_graph.yaml"

python "${ROOT_DIR}/src/pdvi/data/preprocess_anndata.py"
python "${ROOT_DIR}/src/pdvi/data/build_tensor_dataset.py" --seed "${SEED}"
python "${ROOT_DIR}/src/pdvi/graph/compute_tangent.py" --seed "${SEED}"
python "${ROOT_DIR}/src/pdvi/graph/compute_neighbors.py"
python "${ROOT_DIR}/src/pdvi/graph/compute_weight.py" --seed "${SEED}"
python "${ROOT_DIR}/src/pdvi/models/train_graph.py" --config "${CONFIG}" --seed "${SEED}"
