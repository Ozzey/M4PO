#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_DISABLE_DYNAMO="${TORCH_DISABLE_DYNAMO:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export M4PO_DISABLE_TRITON="${M4PO_DISABLE_TRITON:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
PYTHON="${PYTHON:-python}"

RUN_DIR="${RUN_DIR:-artifacts/runs/mock_smoke}"
TOTAL_STEPS="${TOTAL_STEPS:-64}"
EPISODES="${EPISODES:-2}"

set -x
"${PYTHON}" scripts/test_env_mock.py --steps 4
"${PYTHON}" -m m4po.train \
  --config m4po/configs/mock_smoke.yaml \
  --total-steps "${TOTAL_STEPS}" \
  --log-dir "${RUN_DIR}"
"${PYTHON}" -m m4po.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/latest.pt" \
  --episodes "${EPISODES}" \
  --device cpu \
  --out "${RUN_DIR}/benchmark.json" \
  --csv-out "${RUN_DIR}/benchmark.csv" \
  --deterministic
set +x

cat "${RUN_DIR}/benchmark.json"

