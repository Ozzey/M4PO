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

CHECKPOINT="${CHECKPOINT:-${1:-runs/m4po/checkpoints/latest.pt}}"
EPISODES="${EPISODES:-100}"
DEVICE="${DEVICE:-auto}"
OUT="${OUT:-runs/m4po/benchmark.json}"
CSV_OUT="${CSV_OUT:-${OUT%.json}.csv}"

set -x
"${PYTHON}" -m m4po.evaluate \
  --checkpoint "${CHECKPOINT}" \
  --episodes "${EPISODES}" \
  --device "${DEVICE}" \
  --out "${OUT}" \
  --csv-out "${CSV_OUT}" \
  --deterministic
set +x
