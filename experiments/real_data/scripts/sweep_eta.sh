#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP_CONFIG="${1:-${ROOT_DIR}/experiments/real_data/configs/sweep_eta.yaml}"
BASE_CONFIG="${2:-${ROOT_DIR}/experiments/real_data/configs/train_graph.yaml}"
SCRIPT="${ROOT_DIR}/src/pdvi/models/train_graph.py"

OUT="$(python - "${SWEEP_CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
print(cfg.get('output_file', 'experiments/real_data/results/grid_eta_results.txt'))
PY
)"
OUT="${ROOT_DIR}/${OUT}"
mkdir -p "$(dirname "${OUT}")"
: > "${OUT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

readarray -t ETA_MS < <(python - "${SWEEP_CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
vals = cfg.get('eta_m_values', [2e-2, 4e-2, 8e-2, 1e-1])
for v in vals:
    print(v)
PY
)

readarray -t ETA_SS < <(python - "${SWEEP_CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
vals = cfg.get('eta_s_values', [2e-1, 4e-1, 8e-1, 1])
for v in vals:
    print(v)
PY
)

for em in "${ETA_MS[@]}"; do
  for es in "${ETA_SS[@]}"; do
    echo "============================================================" >> "${OUT}"
    echo "[$(timestamp)] eta_m=${em}, eta_s=${es}" >> "${OUT}"
    echo "============================================================" >> "${OUT}"

    tmp_cfg="$(mktemp -t train_graph_cfg_XXXXXX.yaml)"
    python - "${BASE_CONFIG}" "${tmp_cfg}" "${em}" "${es}" <<'PY'
import sys
import yaml

base_cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
eta_m = float(sys.argv[3])
eta_s = float(sys.argv[4])
base_cfg.setdefault('optimization', {})['eta_m'] = eta_m
base_cfg.setdefault('optimization', {})['eta_s'] = eta_s
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    yaml.safe_dump(base_cfg, f, sort_keys=False)
PY

    tmp_log="$(mktemp -t run_eta_XXXXXX.log)"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${SCRIPT}" --config "${tmp_cfg}" > "${tmp_log}" 2>&1 || true

    echo "--- last 10 iter-prints ---" >> "${OUT}"
    grep -E '^iter[[:space:]]' "${tmp_log}" | tail -n 10 >> "${OUT}" || {
      echo "(No lines matched '^iter ' in log.)" >> "${OUT}"
    }
    echo "" >> "${OUT}"

    rm -f "${tmp_log}" "${tmp_cfg}"
  done
done

echo "Done. Results saved to: ${OUT}"
