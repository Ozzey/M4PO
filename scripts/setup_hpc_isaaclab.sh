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
    if [[ "${allocated_host}" == "${current_host}" ]]; then
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
echo "Running setup in Slurm job ${SLURM_JOB_ID} on $(hostname)."

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
MINIFORGE_ROOT="${M4PO_MINIFORGE_ROOT:-${HPC_ROOT}/miniforge3}"
ENV_ROOT="${M4PO_ENV_ROOT:-${HPC_ROOT}/envs/env_isaaclab}"
ISAACLAB_ROOT="${M4PO_ISAACLAB_ROOT:-${HPC_ROOT}/src/IsaacLab-v2.3.0}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
ISAACLAB_COMMIT="3c6e67bb5c7ada942a6d1884ab69338f57596f77"
MINIFORGE_VERSION="${M4PO_MINIFORGE_VERSION:-26.7.2-0}"

mkdir -p \
  "${HPC_ROOT}/envs" \
  "${HPC_ROOT}/src" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/tmp"

export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"

if [[ ! -x "${MINIFORGE_ROOT}/bin/conda" ]]; then
  installer_name="Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
  installer="${CACHE_ROOT}/${installer_name}"
  checksum="${installer}.sha256"
  release_url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}"
  curl --fail --location --retry 5 \
    --output "${installer}" \
    "${release_url}/${installer_name}"
  curl --fail --location --retry 5 \
    --output "${checksum}" \
    "${release_url}/${installer_name}.sha256"
  (
    cd "${CACHE_ROOT}"
    sha256sum --check "$(basename "${checksum}")"
  )
  bash "${installer}" -b -p "${MINIFORGE_ROOT}"
fi

# shellcheck disable=SC1091
source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  conda create --yes --prefix "${ENV_ROOT}" python=3.11 pip
fi
conda activate "${ENV_ROOT}"

python -m pip install --upgrade \
  pip==26.1.2 \
  setuptools==80.9.0 \
  wheel==0.42.0
python -m pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.0 gymnasium==1.2.0 pytest
# ``flatdict`` still imports ``pkg_resources`` while building and fails with
# newer isolated setuptools releases.  Prebuild it with the validated local
# toolchain so IsaacLab sees an already-satisfied dependency.
python -m pip install flatdict==4.0.1 --no-build-isolation

if [[ ! -d "${ISAACLAB_ROOT}/.git" ]]; then
  git clone \
    --branch v2.3.0 \
    --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git \
    "${ISAACLAB_ROOT}"
fi

actual_commit="$(git -C "${ISAACLAB_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${ISAACLAB_COMMIT}" ]]; then
  echo "Expected IsaacLab ${ISAACLAB_COMMIT}, found ${actual_commit}" >&2
  exit 1
fi

for extension in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  python -m pip install --editable "${ISAACLAB_ROOT}/source/${extension}"
done

python -m pip install --editable "${REPO_ROOT}" --no-deps
# IsaacLab's broad transitive constraints otherwise select newer releases that
# conflict with the exact click/typing versions required by Isaac Sim 5.1.
# Keep this compatibility set aligned with the validated local environment.
python -m pip install \
  transformers==4.57.6 \
  huggingface-hub==0.36.2 \
  onnx==1.21.0 \
  click==8.1.7 \
  typing_extensions==4.12.2
python -m pip check

python - <<'PY'
from importlib.metadata import version
import sys

for package in ("isaacsim", "isaaclab", "isaaclab_tasks", "torch", "numpy", "gymnasium", "m4po"):
    print(f"{package}={version(package)}")
print(f"python={sys.version.split()[0]}")
PY

(
  cd "${REPO_ROOT}"
  python -m pytest -q
)
