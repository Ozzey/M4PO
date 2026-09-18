#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

assert_slurm_compute_node() {
  if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to train outside a Slurm allocation." >&2
    exit 1
  fi

  local current_host allocated_host matched=0
  current_host="$(hostname -s)"
  while IFS= read -r allocated_host; do
    if [[ "${allocated_host%%.*}" == "${current_host}" ]]; then
      matched=1
      break
    fi
  done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  if ((matched == 0)); then
    echo "Refusing to train on ${current_host}; it is not in ${SLURM_JOB_NODELIST}." >&2
    exit 1
  fi
}

assert_slurm_compute_node
echo "Running MMBench in Slurm job ${SLURM_JOB_ID} on $(hostname)."

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
CONFIG="${CONFIG:-m4po/configs/mmbench_pilot.yaml}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-${HPC_ROOT}/runs/mmbench_seed${SEED}_${SLURM_JOB_ID}}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  echo "This benchmark launcher starts fresh; RESUME_CHECKPOINT must be unset." >&2
  exit 1
fi
if [[ -e "${RUN_DIR}/checkpoints/latest.pt" || -e "${RUN_DIR}/metrics.jsonl" ]]; then
  echo "Refusing to overwrite an existing run: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/xdg"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_DISABLE_DYNAMO=1
export TORCHDYNAMO_DISABLE=1
export M4PO_DISABLE_TRITON=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="${CACHE_ROOT}/tmp"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export M4PO_NEWT_ROOT="${M4PO_NEWT_ROOT:-${REPO_ROOT}/external/newt}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-${MANISKILL_ASSET_DIR:-${HPC_ROOT}/assets/mmbench/.maniskill}}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys
import torch
from m4po.common.config import M4POConfig

cfg = M4POConfig.from_yaml(sys.argv[1])
if cfg.env != "mmbench" or cfg.learning_mode != "off_policy" or cfg.resume_checkpoint:
    raise SystemExit("MMBench launch requires fresh off-policy configuration.")
if not torch.cuda.is_available():
    raise SystemExit("A visible allocated CUDA device is required.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: {sys.argv[1]}")
PY

args=(
  -m m4po.train
  --config "${CONFIG}"
  --seed "${SEED}"
  --device cuda
  --log-dir "${RUN_DIR}"
)
if [[ -n "${TOTAL_STEPS:-}" ]]; then
  args+=(--total-steps "${TOTAL_STEPS}")
fi
"${PYTHON}" "${args[@]}"
"${PYTHON}" -m m4po.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/latest.pt" \
  --episodes "${EVAL_EPISODES}" \
  --device cuda \
  --out "${RUN_DIR}/benchmark.json" \
  --csv-out "${RUN_DIR}/benchmark.csv"
echo "Completed training and frozen-checkpoint evaluation: ${RUN_DIR}"
