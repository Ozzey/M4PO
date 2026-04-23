#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../..")"
ENV_NAME="${1:-env_isaacsim}"

source /home/aditya/miniconda3/etc/profile.d/conda.sh
export PATH=/home/aditya/miniconda3/bin:$PATH

cd "${REPO_ROOT}"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

./isaaclab.sh --conda "${ENV_NAME}"

conda activate "${ENV_NAME}"

# Install Isaac Lab packages without extra RL frameworks.
./isaaclab.sh -i none

# Install the lightweight GR00T client pieces for policy-server mode.
python -m pip install pyzmq msgpack msgpack-numpy
python -m pip install --no-deps --ignore-requires-python -e "${REPO_ROOT}/external/Isaac-GR00T"

python - <<'PY'
from gr00t.policy.server_client import PolicyClient

print(f"GR00T client import OK: {PolicyClient.__name__}")
PY

cat <<EOF

Setup complete.

Next:
  1. conda activate ${ENV_NAME}
  2. Start a GR00T policy server from a separate Python 3.10 GR00T environment
  3. Run:
     ./isaaclab.sh -p scripts/imitation_learning/gr00t/play.py \\
       --task Isaac-PickPlace-G1-InspireFTP-GR00T-Abs-v0 \\
       --server_host 127.0.0.1 \\
       --server_port 5555 \\
       --instruction "pick up the steering wheel and place it on the table" \\
       --device cuda:0

EOF
