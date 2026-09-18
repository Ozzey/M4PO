#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench-eval
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "Refusing to evaluate outside a Slurm allocation." >&2
  exit 1
fi
current_host="$(hostname -s)"
matched=0
while IFS= read -r allocated_host; do
  if [[ "${allocated_host%%.*}" == "${current_host}" ]]; then matched=1; break; fi
done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((matched == 0)); then
  echo "Refusing to evaluate on ${current_host}: not an allocated node." >&2
  exit 1
fi

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
: "${RUN_DIR:?Set RUN_DIR to the completed full-benchmark run.}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SEED="${SEED:-0}"
M4PO_PARTITION="${M4PO_PARTITION:-${SLURM_JOB_PARTITION:-ws-ia}}"
exec 9>"${RUN_DIR}/.workflow.lock"
if ! flock -n 9; then
  echo "Another training/evaluation process owns ${RUN_DIR}." >&2
  exit 1
fi
if [[ -e "${RUN_DIR}/benchmark.json" || -e "${RUN_DIR}/comparison.json" ]]; then
  echo "Refusing to overwrite existing final benchmark/comparison results." >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/xdg"
cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_DISABLE_DYNAMO=1
export TORCHDYNAMO_DISABLE=1
export M4PO_DISABLE_TRITON=1
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
export M4PO_HPC_ROOT="${HPC_ROOT}"
export M4PO_REPO_ROOT="${REPO_ROOT}"
export M4PO_PYTHON="${PYTHON}"
export RUN_DIR
export M4PO_ENABLE_WANDB="${M4PO_ENABLE_WANDB:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-adityanarendra5}"
export WANDB_PROJECT="${WANDB_PROJECT:-m4po-mmbench}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-m4po-mmbench-all-seed${SEED}-$(basename "${RUN_DIR}")}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-M4PO MMBench all 200 - seed ${SEED}}"
export WANDB_GROUP="${WANDB_GROUP:-mmbench-all-from-scratch}"
export M4PO_WANDB_CHECKPOINT="${RUN_DIR}/checkpoints/final_model.pt"

CHECKPOINT="$("${PYTHON}" - "${RUN_DIR}" "${EVAL_EPISODES}" <<'PY'
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import torch
from m4po.common.config import M4POConfig
from m4po.mmbench_preflight import verify_slurm_allocation

verify_slurm_allocation()
if not torch.cuda.is_available() or int(sys.argv[2]) != 10:
    raise SystemExit("Final benchmark requires allocated CUDA and exactly 10 episodes per task.")
run = Path(sys.argv[1])
source_root = Path.cwd()
package_hash = hashlib.sha256()
sources = [source for source in (source_root / "m4po").rglob("*") if source.suffix in (".py", ".yaml", ".yml")]
sources += [source_root / "scripts" / name for name in ("train_mmbench_full_hpc.sbatch", "evaluate_mmbench_hpc.sbatch", "preflight_mmbench_hpc.sbatch", "smoke_mmbench_hpc.sbatch", "pretrain_mmbench_hpc.sbatch", "smoke_mmbench_demos_hpc.sbatch", "sync_wandb_hpc.sbatch")]
for source in sorted(sources):
    package_hash.update(source.relative_to(source_root).as_posix().encode())
    package_hash.update(b"\0")
    package_hash.update(source.read_bytes())
    package_hash.update(b"\0")
manifest = json.loads((run / "launch_manifest.json").read_text())
if manifest.get("identity", {}).get("m4po_sources_sha256") != package_hash.hexdigest():
    raise SystemExit("M4PO implementation changed since training began; refusing mixed-implementation evaluation.")
latest = run / "checkpoints/latest.pt"
status = json.loads((run / "checkpoints/latest_status.json").read_text())
checkpoint_stat = latest.stat()
if status.get("checkpoint_size") != checkpoint_stat.st_size or status.get("checkpoint_mtime_ns") != checkpoint_stat.st_mtime_ns:
    raise SystemExit("Latest checkpoint status is stale; refusing final evaluation.")
if (
    status.get("training_complete") is not True or status.get("step") != 100_000_000
    or status.get("last_eval_step") != 100_000_000
    or status.get("pending_evaluation") is not False
):
    raise SystemExit("The final transition budget, pending optimizer updates, and scheduled evaluations must be complete.")
checkpoint = run / "checkpoints/final_model.pt"
# A same-step archive can predate the final transition's pending optimizer work.
# Derive the evaluation artifact from the verified complete resumable checkpoint.
payload = torch.load(latest, map_location="cpu", weights_only=False)
cfg = M4POConfig.from_checkpoint_dict(payload["cfg"])
if (
    payload.get("step") != 100_000_000 or cfg.total_steps != 100_000_000
    or cfg.env != "mmbench" or cfg.learning_mode != "off_policy"
    or cfg.num_tasks != 200 or cfg.mmbench_sampling != "fixed"
    or payload.get("extra", {}).get("training_complete") is not True
    or payload.get("extra", {}).get("update_count") != status.get("update_count")
    or payload.get("extra", {}).get("last_eval_step") != 100_000_000
    or payload.get("extra", {}).get("pending_evaluation") is not False
):
    raise SystemExit("Final evaluation requires completed all-200-task off-policy weights at exactly 100M transitions.")
# Publish only the final weights/optimizer metadata, never multi-GB replay,
# so evaluation and the W&B artifact reference precisely the same policy.
payload["extra"].pop("replay_state", None)
payload["extra"]["replay_saved"] = False
payload["extra"]["resume_rebuilds_replay"] = True
descriptor, temporary = tempfile.mkstemp(prefix=".final_weights.", suffix=".tmp", dir=checkpoint.parent)
os.close(descriptor)
try:
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(checkpoint)
PY
)"
sidecar_job=""
if [[ "${M4PO_ENABLE_WANDB}" == "1" ]]; then
  if sidecar_response="$(sbatch --parsable --partition="${M4PO_PARTITION}" \
    --cpus-per-task=2 --mem=4G --gres=none --time=24:00:00 \
    --chdir="${REPO_ROOT}" --output="${RUN_DIR}/slurm-wandb-%j.out" \
    --export="ALL,TRAINING_JOB_ID=${SLURM_JOB_ID},EVALUATION_JOB_ID=${SLURM_JOB_ID}" \
    "${REPO_ROOT}/scripts/sync_wandb_hpc.sbatch")"; then
    sidecar_job="${sidecar_response%%;*}"
  else
    echo "W&B sidecar submission failed; final benchmark files remain available locally." >&2
  fi
fi
echo "Final evaluation uses ${CHECKPOINT}; W&B sidecar ${sidecar_job:-not submitted}."
"${PYTHON}" -m m4po.evaluate \
  --checkpoint "${CHECKPOINT}" --episodes "${EVAL_EPISODES}" --device cuda \
  --out "${RUN_DIR}/benchmark.json" --csv-out "${RUN_DIR}/benchmark.csv"
"${PYTHON}" -m m4po.compare_mmbench "${RUN_DIR}/benchmark.json" \
  --curves "${RUN_DIR}/metrics.jsonl" \
  --out-prefix "${RUN_DIR}/comparison"
echo "Completed frozen all-200-task benchmark and comparison report: ${RUN_DIR}"
