#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench-preflight
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "Refusing to preflight outside a Slurm allocation." >&2
  exit 1
fi
current_host="$(hostname -s)"
matched=0
while IFS= read -r allocated_host; do
  if [[ "${allocated_host%%.*}" == "${current_host}" ]]; then matched=1; break; fi
done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((matched == 0)); then
  echo "Refusing to preflight on ${current_host}: not an allocated node." >&2
  exit 1
fi

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
CONFIG="${CONFIG:-m4po/configs/mmbench_all.yaml}"
RUN_DIR="${RUN_DIR:-${HPC_ROOT}/runs/mmbench_preflight_${SLURM_JOB_ID}}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${RUN_DIR}/preflight.json}"

mkdir -p "${RUN_DIR}" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/xdg"
cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-^openib}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export TMPDIR="${CACHE_ROOT}/tmp"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export M4PO_NEWT_ROOT="${M4PO_NEWT_ROOT:-${REPO_ROOT}/external/newt}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-${MANISKILL_ASSET_DIR:-${HPC_ROOT}/assets/mmbench/.maniskill}}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

echo "Preflight job ${SLURM_JOB_ID} on ${current_host}; report ${PREFLIGHT_REPORT}"
"${PYTHON}" -m pytest -q
"${PYTHON}" -m m4po.mmbench_preflight \
  --config "${CONFIG}" \
  --out "${PREFLIGHT_REPORT}" \
  --require-slurm \
  --workers 2 \
  --timeout 180
