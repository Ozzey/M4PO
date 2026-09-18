#!/usr/bin/env bash
#SBATCH --job-name=m4po-wandb
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

assert_slurm_compute_node() {
  if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to sync W&B outside a Slurm allocation." >&2
    exit 1
  fi

  local current_host allocated_host matched=0
  current_host="$(hostname -s)"
  while IFS= read -r allocated_host; do
    if [[ "${allocated_host}" == "${current_host}" ]]; then
      matched=1
      break
    fi
  done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  if ((matched == 0)); then
    echo "Refusing to sync on ${current_host}; it is not in ${SLURM_JOB_NODELIST}." >&2
    exit 1
  fi
}

assert_slurm_compute_node
echo "Running W&B sidecar in Slurm job ${SLURM_JOB_ID} on $(hostname)."

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
BASE_PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_isaaclab/bin/python}"
WANDB_ENV="${M4PO_WANDB_ENV:-${HPC_ROOT}/envs/wandb}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
RUN_DIR="${RUN_DIR:-${HPC_ROOT}/runs/isaaclab_forge_seed0}"
TRAINING_JOB_ID="${TRAINING_JOB_ID:-197854}"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID:-197858}"
WANDB_ENTITY="${WANDB_ENTITY:-adityanarendra5}"
WANDB_PROJECT="${WANDB_PROJECT:-m4po-isaaclab-forge}"
WANDB_RUN_ID="${WANDB_RUN_ID:-m4po-forge-seed0-${TRAINING_JOB_ID}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-M4PO Forge seed 0 - Slurm ${TRAINING_JOB_ID}}"
WANDB_GROUP="${WANDB_GROUP:-forge-multitask}"
CHECKPOINT_PATH="${M4PO_WANDB_CHECKPOINT:-${RUN_DIR}/checkpoints/latest.pt}"

mkdir -p \
  "${HPC_ROOT}/envs" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/tmp" \
  "${CACHE_ROOT}/wandb" \
  "${RUN_DIR}/wandb"

export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb"
export PYTHONUNBUFFERED=1

if [[ ! -x "${WANDB_ENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv "${WANDB_ENV}"
fi
WANDB_PYTHON="${WANDB_ENV}/bin/python"
if ! "${WANDB_PYTHON}" -c 'import wandb; assert wandb.__version__ == "0.28.1"' \
  >/dev/null 2>&1; then
  (
    cd "${CACHE_ROOT}/tmp"
    env -u PYTHONPATH \
      "${WANDB_PYTHON}" -m pip install wandb==0.28.1 PyYAML==6.0.2
  )
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  -m m4po.wandb_sync
  --metrics "${RUN_DIR}/metrics.jsonl"
  --cursor "${RUN_DIR}/wandb_sync_cursor.json"
  --url-file "${RUN_DIR}/wandb_url.txt"
  --config "${RUN_DIR}/config_resolved.yaml"
  --benchmark "${RUN_DIR}/benchmark.json"
  --checkpoint "${CHECKPOINT_PATH}"
  --training-job-id "${TRAINING_JOB_ID}"
  --entity "${WANDB_ENTITY}"
  --project "${WANDB_PROJECT}"
  --run-id "${WANDB_RUN_ID}"
  --name "${WANDB_RUN_NAME}"
  --group "${WANDB_GROUP}"
  --poll-seconds 15
)
if [[ -n "${EVALUATION_JOB_ID}" ]]; then
  args+=(--evaluation-job-id "${EVALUATION_JOB_ID}")
fi

"${WANDB_PYTHON}" "${args[@]}"
