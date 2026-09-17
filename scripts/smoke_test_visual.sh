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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-artifacts/runs/mock_visual_smoke}"

set -x
"${PYTHON}" -m m4po.train \
  --config m4po/configs/mock_visual_smoke.yaml \
  --log-dir "${RUN_DIR}"
"${PYTHON}" -m m4po.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/latest.pt" \
  --episodes 1 \
  --device cpu \
  --out "${RUN_DIR}/benchmark.json"
set +x

