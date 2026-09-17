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
PYTHON="${PYTHON:-python}"

CONFIG="${CONFIG:-m4po/config.yaml}"
ENV_NAME="${ENV_NAME:-mock}"
TOTAL_STEPS="${TOTAL_STEPS:-1048576}"
NUM_ENVS="${NUM_ENVS:-8}"
DEVICE="${DEVICE:-auto}"
LOG_DIR="${LOG_DIR:-runs/m4po}"
SEED="${SEED:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

CMD=(
  "${PYTHON}" -m m4po.train
  --config "${CONFIG}"
  --env "${ENV_NAME}"
  --seed "${SEED}"
  --total-steps "${TOTAL_STEPS}"
  --num-envs "${NUM_ENVS}"
  --device "${DEVICE}"
  --log-dir "${LOG_DIR}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

set -x
"${CMD[@]}"
set +x

