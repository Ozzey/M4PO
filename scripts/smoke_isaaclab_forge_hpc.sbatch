#!/usr/bin/env bash
#SBATCH --job-name=m4po-smoke
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

assert_slurm_compute_node() {
  if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to run the smoke test outside a Slurm allocation." >&2
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
    echo "Refusing to run on ${current_host}; it is not in ${SLURM_JOB_NODELIST}." >&2
    exit 1
  fi
}

assert_slurm_compute_node
if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
  echo "Set OMNI_KIT_ACCEPT_EULA=YES after accepting NVIDIA's Omniverse EULA." >&2
  exit 1
fi
echo "Running smoke test in Slurm job ${SLURM_JOB_ID} on $(hostname)."

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_isaaclab/bin/python}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
RUN_DIR="${RUN_DIR:-${HPC_ROOT}/runs/isaaclab_forge_smoke_${SLURM_JOB_ID}}"

mkdir -p "${RUN_DIR}" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/xdg"
cd "${REPO_ROOT}"

export ISAACLAB_PYTHON="${PYTHON}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TORCH_DISABLE_DYNAMO=1
export TORCHDYNAMO_DISABLE=1
export M4PO_DISABLE_TRITON=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="${CACHE_ROOT}/tmp"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

RUN_DIR="${RUN_DIR}" \
TOTAL_STEPS=768 \
EPISODES=1 \
bash scripts/smoke_test_isaaclab_forge.sh
