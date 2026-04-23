#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/..")"
MODEL_DIR="${1:-${SCRIPT_DIR}/assets/model/GN1x-Tuned-Arena-G1-Loco-Manipulation}"
HOST="${GROOT_HOST:-0.0.0.0}"
PORT="${GROOT_PORT:-5555}"
DEVICE="${GROOT_DEVICE:-cuda}"
EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-new_embodiment}"
MODEL_TYPE="$(python -c 'import json, pathlib, sys; cfg = pathlib.Path(sys.argv[1]) / "config.json"; print(json.loads(cfg.read_text()).get("model_type", "") if cfg.exists() else "")' "${MODEL_DIR}")"

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Model directory not found: ${MODEL_DIR}"
  echo "Run: python ${SCRIPT_DIR}/download_model.py"
  exit 1
fi

if [ "${MODEL_TYPE}" = "gr00t_n1_5" ]; then
  echo "Detected GR00T N1.5 checkpoint; using the local N1.5 compatibility server."
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
  python "${SCRIPT_DIR}/legacy_server.py" \
    --model-path "${MODEL_DIR}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --device "${DEVICE}" \
    --host "${HOST}" \
    --port "${PORT}"
else
  python "${REPO_ROOT}/isaaclab/external/Isaac-GR00T/gr00t/eval/run_gr00t_server.py" \
    --model-path "${MODEL_DIR}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --device "${DEVICE}" \
    --host "${HOST}" \
    --port "${PORT}"
fi
