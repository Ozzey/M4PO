#!/usr/bin/env bash
set -euo pipefail

assert_slurm_compute_node() {
  if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to install outside a Slurm allocation." >&2
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
    echo "Refusing to install on ${current_host}; it is not in ${SLURM_JOB_NODELIST}." >&2
    exit 1
  fi
}

assert_slurm_compute_node
INSTALL_MODE="${1:-pilot}"
if [[ "${INSTALL_MODE}" != "pilot" && "${INSTALL_MODE}" != "all" ]]; then
  echo "Usage: $0 [pilot|all]" >&2
  exit 1
fi
echo "Running MMBench ${INSTALL_MODE} setup in Slurm job ${SLURM_JOB_ID} on $(hostname)."

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
ENV_ROOT="${M4PO_MMBENCH_ENV_ROOT:-${HPC_ROOT}/envs/env_mmbench}"
BASE_PYTHON="${M4PO_BASE_PYTHON:-${HPC_ROOT}/envs/env_isaaclab/bin/python}"
NEWT_ROOT="${M4PO_NEWT_ROOT:-${REPO_ROOT}/external/newt}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
ASSET_ROOT="${M4PO_MMBENCH_ASSET_ROOT:-${HPC_ROOT}/assets/mmbench}"
NEWT_COMMIT="1d3fc058b81ddf8d36a5457c29ea407dd374c1b8"
METAWORLD_COMMIT="22904d1f65afe920be4325d482808c385f4c0c38"

mkdir -p "${HPC_ROOT}/envs" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/downloads"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export PYTHONNOUSERSITE=1
export M4PO_NEWT_ROOT="${NEWT_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ ! -d "${NEWT_ROOT}/.git" && ! -f "${NEWT_ROOT}/.git" ]]; then
  if [[ -e "${NEWT_ROOT}" ]]; then
    echo "Expected a pinned Newt git checkout at ${NEWT_ROOT}; refusing to overwrite it." >&2
    exit 1
  fi
  git clone --no-checkout https://github.com/nicklashansen/newt.git "${NEWT_ROOT}"
  git -C "${NEWT_ROOT}" checkout --detach "${NEWT_COMMIT}"
fi
actual_commit="$(git -C "${NEWT_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${NEWT_COMMIT}" ]]; then
  echo "Expected Newt ${NEWT_COMMIT}, found ${actual_commit}." >&2
  exit 1
fi

# Reuse the existing CUDA/PyTorch installation without changing its packages.
# The benchmark's older Gymnasium/MuJoCo dependencies live only in this venv.
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${ENV_ROOT}"
fi
PYTHON="${ENV_ROOT}/bin/python"
export PATH="${ENV_ROOT}/bin:${PATH}"
"${PYTHON}" - "${ENV_ROOT}" <<'PY'
from pathlib import Path
import sys

expected = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit(f"Refusing to install outside the dedicated MMBench venv: {sys.prefix}")
import torch
print(f"Reusing torch={torch.__version__} from {torch.__file__}")
PY
export PIP_REQUIRE_VIRTUALENV=true

"${PYTHON}" -m pip install \
  gymnasium==0.29.1 \
  dm-control==1.0.34 \
  mujoco==3.3.6 \
  "numpy>=1.26,<2" \
  "PyYAML>=6.0"

if [[ "${INSTALL_MODE}" == "all" ]]; then
  "${PYTHON}" -m pip install swig==4.3.1
  "${PYTHON}" -m pip install \
    "gymnasium[box2d]==0.29.1" \
    pygame==2.6.1 \
    ale-py==0.10.0 \
    mani_skill-nightly==2025.9.19.39 \
    ogbench==1.1.5 \
    robodesk==1.0.0 \
    "git+https://github.com/nicklashansen/metaworld.git@${METAWORLD_COMMIT}"
  export MS_ASSET_DIR="${MS_ASSET_DIR:-${MANISKILL_ASSET_DIR:-${ASSET_ROOT}/.maniskill}}"
  if [[ ! -d "${MS_ASSET_DIR}" ]]; then
    if [[ "${MS_ASSET_DIR}" != "${ASSET_ROOT}/.maniskill" ]]; then
      echo "Custom MS_ASSET_DIR must already contain the downloaded assets." >&2
      exit 1
    fi
    mkdir -p "${ASSET_ROOT}"
    asset_archive="${CACHE_ROOT}/downloads/mmbench-maniskill.tar.gz"
    curl --fail --location --retry 5 \
      --output "${asset_archive}" \
      https://huggingface.co/datasets/nicklashansen/mmbench/resolve/main/maniskill.tar.gz
    "${PYTHON}" - "${asset_archive}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1]) as archive:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit(f"Unsafe archive member: {member.name}")
        if not path.parts or path.parts[0] != ".maniskill":
            raise SystemExit(f"Unexpected asset archive root: {member.name}")
PY
    tar -xzf "${asset_archive}" -C "${ASSET_ROOT}"
  fi
  echo "ALE 0.10.0 bundles Atari ROMs; native-task preflight verifies their availability."
fi

"${PYTHON}" -m pip install --editable "${REPO_ROOT}" --no-deps
"${PYTHON}" - <<'PY'
from importlib.metadata import version
import sys
import gymnasium
import mujoco
import torch

for package in ("torch", "numpy", "gymnasium", "dm-control", "mujoco", "m4po"):
    print(f"{package}={version(package)}")
print(f"python={sys.version.split()[0]}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
echo "MMBench environment ready: ${PYTHON}"
echo "Newt source: ${NEWT_ROOT} @ ${NEWT_COMMIT}"
