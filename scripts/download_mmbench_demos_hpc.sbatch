#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench-demos
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=none
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "Refusing to install/download outside a Slurm allocation." >&2
  exit 1
fi
HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
DEMO_DATA_DIR="${DEMO_DATA_DIR:-${HPC_ROOT}/datasets/mmbench-a59d457}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="/scratch/${USER}/m4po-cache/pip"
export TMPDIR="/scratch/${USER}/m4po-cache/tmp"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
cd "${REPO_ROOT}"
"${PYTHON}" -c 'from m4po.mmbench_preflight import verify_slurm_allocation; print(verify_slurm_allocation())'
# TensorDict is needed only to deserialize the released demonstration shards.
# The installed PyTorch 2.7 remains unchanged in the isolated MMBench venv.
"${PYTHON}" -m pip install tensordict==0.8.3
"${PYTHON}" -m m4po.download_mmbench_demos \
  --out-dir "${DEMO_DATA_DIR}" --workers 4 --require-slurm
