#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench-demo-smoke
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "Refusing to pretrain outside a Slurm allocation." >&2
  exit 1
fi
HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
DEMO_DATA_DIR="${DEMO_DATA_DIR:-${HPC_ROOT}/datasets/mmbench-a59d457}"
DEMO_SMOKE_RUN_DIR="${DEMO_SMOKE_RUN_DIR:-${HPC_ROOT}/runs/mmbench_demo_smoke_${SLURM_JOB_ID}}"
: "${PREFLIGHT_REPORT:?Set the successful 200-task native preflight report.}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_DISABLE_DYNAMO=1 TORCHDYNAMO_DISABLE=1 M4PO_DISABLE_TRITON=1
cd "${REPO_ROOT}"
"${PYTHON}" -m m4po.pretrain_demonstrations \
  --config m4po/configs/mmbench_all.yaml --data-dir "${DEMO_DATA_DIR}" \
  --preflight-report "${PREFLIGHT_REPORT}" --updates 100 \
  --log-dir "${DEMO_SMOKE_RUN_DIR}" --device cuda --log-every 10 \
  --require-slurm
"${PYTHON}" - "${DEMO_SMOKE_RUN_DIR}" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np
import torch
from m4po.common.buffer import ObservationSpec
from m4po.common.config import M4POConfig
from m4po.common.initialization import load_demonstration_initialization
from m4po.envs.mmbench_env import load_mmbench_catalog
from m4po.mmbench_preflight import verify_slurm_allocation

run = Path(sys.argv[1])
cfg = M4POConfig.from_yaml("m4po/configs/mmbench_pretrained.yaml")
cfg.init_checkpoint = str(run / "checkpoints/pretrained_model.pt")
metadata, _, _ = load_mmbench_catalog(cfg.mmbench_root)
contexts = np.asarray([metadata[task]["text_embedding"] for task in cfg.task_names], dtype=np.float32)
agent, provenance = load_demonstration_initialization(
    cfg.init_checkpoint, cfg, ObservationSpec((0, 0, 0), 0, 128),
    16, contexts, torch.device("cuda"),
)
demo = provenance["demonstration_pretraining"]
if demo["completed_updates"] != 100 or demo["task_count"] != 200:
    raise SystemExit("Demo smoke must complete 100 real updates on the full dataset.")
result = {
    "passed": True, "allocation": verify_slurm_allocation(),
    "purpose": "100-update full-dataset offline and warm-start compatibility smoke, not benchmark performance",
    "demonstration_pretraining": demo,
    "initialization": provenance,
    "online_seed_steps": cfg.seed_steps,
    "online_replay_warmup_updates": cfg.pretrain_updates,
}
(run / "smoke_passed.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": True, "tasks": 200, "updates": 100, "run": str(run)}))
PY
