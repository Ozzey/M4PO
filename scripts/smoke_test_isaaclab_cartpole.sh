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

CONFIG="${CONFIG:-m4po/configs/isaaclab_cartpole_smoke.yaml}"
RUN_DIR="${RUN_DIR:-artifacts/runs/isaaclab_cartpole_smoke}"
TOTAL_STEPS="${TOTAL_STEPS:-512}"
EPISODES="${EPISODES:-1}"

set -x
"${ISAACLAB_PYTHON}" -m m4po.train \
  --config "${CONFIG}" \
  --total-steps "${TOTAL_STEPS}" \
  --log-dir "${RUN_DIR}"

# Evaluation starts only after training has reaped its simulator child. Both
# Cartpole registrations are visited sequentially with frozen weights.
"${ISAACLAB_PYTHON}" -m m4po.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/latest.pt" \
  --episodes "${EPISODES}" \
  --device cuda \
  --out "${RUN_DIR}/benchmark.json" \
  --csv-out "${RUN_DIR}/benchmark.csv" \
  --deterministic
set +x

"${ISAACLAB_PYTHON}" - "${RUN_DIR}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
records = [
    json.loads(line)
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
training = [record for record in records if record.get("phase") == "on_policy_training"]
if not training or int(training[-1].get("episodes_completed", 0)) < 64:
    raise SystemExit("Cartpole smoke did not complete at least one episode per environment and rollout")

benchmark = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
expected_tasks = {"Isaac-Cartpole-Direct-v0", "Isaac-Cartpole-v0"}
if set(benchmark.get("per_task", {})) != expected_tasks:
    raise SystemExit("Frozen evaluation did not cover both Cartpole registrations")
for label, record in benchmark.get("per_pair", {}).items():
    lengths = record.get("lengths", [])
    successes = record.get("successes", [])
    if not lengths or len(lengths) != len(successes):
        raise SystemExit(f"Malformed frozen-evaluation episodes for {label}")
    if any(not 1 <= int(length) <= 8 for length in lengths):
        raise SystemExit(f"Episode exceeded the configured eight-step horizon for {label}")
    if any(bool(success) and int(length) != 8 for length, success in zip(lengths, successes, strict=True)):
        raise SystemExit(f"A successful episode did not reach the horizon for {label}")
    if not math.isfinite(float(record["return_mean"])):
        raise SystemExit(f"Non-finite frozen return for {label}")
print("Cartpole smoke assertions passed for both task registrations.")
PY

cat "${RUN_DIR}/benchmark.json"
