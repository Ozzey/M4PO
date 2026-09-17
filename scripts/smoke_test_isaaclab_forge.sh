#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/aditya/miniconda3/envs/env_isaaclab/bin/python}"
if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
  echo "IsaacLab Python was not found at ${ISAACLAB_PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_DISABLE_DYNAMO="${TORCH_DISABLE_DYNAMO:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export M4PO_DISABLE_TRITON="${M4PO_DISABLE_TRITON:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CONFIG="${CONFIG:-m4po/configs/isaaclab_forge_smoke.yaml}"
RUN_DIR="${RUN_DIR:-artifacts/runs/isaaclab_forge_smoke}"
TOTAL_STEPS="${TOTAL_STEPS:-768}"
EPISODES="${EPISODES:-1}"

set -x
"${ISAACLAB_PYTHON}" -m m4po.train \
  --config "${CONFIG}" \
  --total-steps "${TOTAL_STEPS}" \
  --log-dir "${RUN_DIR}"

# Evaluation runs only after training and its simulator child exit. The adapter
# evaluates each task sequentially in a fresh, isolated Isaac Sim child.
"${ISAACLAB_PYTHON}" -m m4po.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/latest.pt" \
  --episodes "${EPISODES}" \
  --device cuda \
  --out "${RUN_DIR}/benchmark.json" \
  --csv-out "${RUN_DIR}/benchmark.csv" \
  --deterministic
set +x

cat "${RUN_DIR}/benchmark.json"
