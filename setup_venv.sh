#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
FRESH_VENV="${FRESH_VENV:-1}"

if [[ "${FRESH_VENV}" == "1" && -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

set -x
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python - <<'PY'
import importlib.util

for name in ["torch", "numpy", "yaml", "m4po"]:
    available = importlib.util.find_spec(name) is not None
    print(f"{name}: {'ok' if available else 'missing'}")
PY
set +x

