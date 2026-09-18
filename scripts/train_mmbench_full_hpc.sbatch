#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench-full
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "Refusing to train outside a Slurm allocation." >&2
  exit 1
fi
current_host="$(hostname -s)"
matched=0
while IFS= read -r allocated_host; do
  if [[ "${allocated_host%%.*}" == "${current_host}" ]]; then matched=1; break; fi
done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((matched == 0)); then
  echo "Refusing to train on ${current_host}: not an allocated node." >&2
  exit 1
fi

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
CONFIG="${CONFIG:-m4po/configs/mmbench_all.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-${HPC_ROOT}/runs/mmbench_all_seed${SEED}_${SLURM_JOB_ID}}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${RUN_DIR}/preflight.json}"
: "${JOINT_SMOKE_REPORT:?Set JOINT_SMOKE_REPORT to the successful all-200-task joint smoke report.}"
CONTINUE_TRAINING="${CONTINUE_TRAINING:-0}"
M4PO_CHUNK_INDEX="${M4PO_CHUNK_INDEX:-1}"
MAX_CHUNKS="${MAX_CHUNKS:-128}"
MAX_WALL_TIME_SECONDS="${MAX_WALL_TIME_SECONDS:-79200}"
M4PO_PARTITION="${M4PO_PARTITION:-${SLURM_JOB_PARTITION:-ws-ia}}"
if [[ "${TOTAL_STEPS:-100000000}" != "100000000" || -n "${RESUME_CHECKPOINT:-}" ]]; then
  echo "Full benchmark budget is fixed at 100M; only this run's latest checkpoint may resume." >&2
  exit 1
fi
if [[ "${CONTINUE_TRAINING}" != "0" && "${CONTINUE_TRAINING}" != "1" ]]; then
  echo "CONTINUE_TRAINING must be 0 or 1." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/xdg"
exec 9>"${RUN_DIR}/.workflow.lock"
if ! flock -n 9; then
  echo "Another training/evaluation process owns ${RUN_DIR}." >&2
  exit 1
fi
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
export CONFIG INIT_CHECKPOINT SEED RUN_DIR PREFLIGHT_REPORT JOINT_SMOKE_REPORT MAX_CHUNKS MAX_WALL_TIME_SECONDS M4PO_PARTITION
export M4PO_ENABLE_WANDB="${M4PO_ENABLE_WANDB:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-adityanarendra5}"
export WANDB_PROJECT="${WANDB_PROJECT:-m4po-mmbench}"
campaign_mode="from-scratch"
if [[ -n "${INIT_CHECKPOINT}" ]]; then campaign_mode="demo-pretrained"; fi
export WANDB_RUN_ID="${WANDB_RUN_ID:-m4po-mmbench-${campaign_mode}-seed${SEED}-$(basename "${RUN_DIR}")}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-M4PO MMBench all 200 ${campaign_mode} - seed ${SEED}}"
export WANDB_GROUP="${WANDB_GROUP:-mmbench-all-${campaign_mode}}"
# Interim chunks sync metrics only; final evaluation creates these verified weights.
export M4PO_WANDB_CHECKPOINT="${RUN_DIR}/checkpoints/final_model.pt"
if [[ -e "${M4PO_WANDB_CHECKPOINT}" ]]; then
  echo "Final evaluation weights already exist; refusing to resume completed training." >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" "${PREFLIGHT_REPORT}" "${JOINT_SMOKE_REPORT}" "${RUN_DIR}" \
  "${SEED}" "${CONTINUE_TRAINING}" "${M4PO_CHUNK_INDEX}" \
  "${MAX_CHUNKS}" "${MAX_WALL_TIME_SECONDS}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import torch
from m4po.common.config import M4POConfig
from m4po.envs.mmbench_env import _native_source_digest, _newt_root, load_mmbench_catalog
from m4po.mmbench_preflight import verify_slurm_allocation

config, preflight, joint_smoke, run = map(Path, sys.argv[1:5])
seed, continuing, chunk, maximum, wall_time = map(int, sys.argv[5:])
if seed < 0 or not 1 <= chunk <= maximum <= 128 or not 60 <= wall_time <= 79200:
    raise SystemExit("Require nonnegative seed, 1 <= chunk <= max_chunks <= 128, and 60..79200 seconds per chunk.")
if not continuing and chunk != 1:
    raise SystemExit("Fresh training must begin at chunk 1.")
allocation = verify_slurm_allocation()
cfg = M4POConfig.from_yaml(config)
cfg.init_checkpoint = os.environ.get("INIT_CHECKPOINT") or None
cfg.validate()
if cfg.init_checkpoint:
    if cfg.seed_steps != 0 or cfg.pretrain_updates != 0:
        raise SystemExit("Demonstration-initialized online training must disable random seeding and replay pretraining.")
elif cfg.seed_steps == 0 and cfg.pretrain_updates == 0:
    raise SystemExit("The demonstration-pretrained online configuration requires INIT_CHECKPOINT.")
if (
    cfg.env != "mmbench" or cfg.learning_mode != "off_policy"
    or cfg.num_tasks != 200 or cfg.num_envs != 200
    or cfg.mmbench_sampling != "fixed" or cfg.total_steps != 100_000_000
    or cfg.batch_size != 1024 or cfg.replay_capacity != 10_000_000
    or not cfg.save_replay or not cfg.eval_at_start or cfg.eval_every != 2_000_000
    or cfg.eval_episodes != 2 or cfg.resume_checkpoint
):
    raise SystemExit("Full launcher requires the all-200 fixed-worker, 100M, replay-persistent, periodically evaluated configuration.")
if not torch.cuda.is_available():
    raise SystemExit("An allocated visible CUDA device is required.")
_, task_sets, catalog_hash = load_mmbench_catalog(cfg.mmbench_root)
native_hash = _native_source_digest(_newt_root(cfg.mmbench_root))
report = json.loads(preflight.read_text())
tasks = set(task_sets["soup"])
if (
    report.get("complete") is not True or report.get("full_suite_passed") is not True
    or report.get("passed_task_count") != 200 or report.get("failed_task_count") != 0
    or report.get("allocation", {}).get("verified") is not True
    or report.get("catalog_sha256") != catalog_hash
    or set(report.get("required_tasks", [])) != tasks
    or set(report.get("per_task", {})) != tasks
    or any(
        row.get("status") != "passed" or row.get("catalog_sha256") != catalog_hash
        or row.get("native_sources_sha256") != native_hash
        for row in report.get("per_task", {}).values()
    )
):
    raise SystemExit("All 200 native tasks must pass allocated-node preflight with matching catalog and native source hashes.")
smoke = json.loads(joint_smoke.read_text())
if (
    smoke.get("passed") is not True or smoke.get("tasks") != 200
    or smoke.get("allocation", {}).get("verified") is not True
    or smoke.get("joint_smoke_steps") != 10_000
    or not isinstance(smoke.get("joint_smoke_updates"), int)
    or smoke["joint_smoke_updates"] <= 0
    or smoke.get("production_action_batch_size") != 200
    or smoke.get("production_update_batch_size") != cfg.batch_size
    or smoke.get("production_trainable_parameters", 0) < 10_000_000
    or any(
        not isinstance(smoke.get(key), (int, float))
        or not math.isfinite(smoke[key]) or smoke[key] <= 0
        for key in ("production_action_batch_mean_ms", "production_update_mean_ms", "production_peak_cuda_bytes")
    )
):
    raise SystemExit("Full training requires a passed joint all-200-task 10k-step smoke with nonzero replay updates and the production-sized GPU profile.")
manifest_path = run / "launch_manifest.json"
source_root = Path.cwd()
package_hash = hashlib.sha256()
sources = [source for source in (source_root / "m4po").rglob("*") if source.suffix in (".py", ".yaml", ".yml")]
sources += [source_root / "scripts" / name for name in ("train_mmbench_full_hpc.sbatch", "evaluate_mmbench_hpc.sbatch", "preflight_mmbench_hpc.sbatch", "smoke_mmbench_hpc.sbatch", "pretrain_mmbench_hpc.sbatch", "smoke_mmbench_demos_hpc.sbatch", "sync_wandb_hpc.sbatch")]
for source in sorted(sources):
    package_hash.update(source.relative_to(source_root).as_posix().encode())
    package_hash.update(b"\0")
    package_hash.update(source.read_bytes())
    package_hash.update(b"\0")
init_hash = None
if cfg.init_checkpoint:
    digest = hashlib.sha256()
    with Path(cfg.init_checkpoint).open("rb") as file:
        for data in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(data)
    init_hash = digest.hexdigest()
identity = {
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "catalog_sha256": catalog_hash,
    "native_sources_sha256": native_hash,
    "m4po_sources_sha256": package_hash.hexdigest(),
    "seed": seed,
    "total_steps": 100_000_000,
    "newt_root": str(_newt_root(cfg.mmbench_root)),
    "preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
    "joint_smoke_sha256": hashlib.sha256(joint_smoke.read_bytes()).hexdigest(),
    "init_checkpoint_sha256": init_hash,
    "init_checkpoint": str(Path(cfg.init_checkpoint).resolve()) if cfg.init_checkpoint else None,
}
if continuing:
    if not (run / "checkpoints/latest.pt").is_file() or not (run / "metrics.jsonl").is_file():
        raise SystemExit("Continuation requires this run's latest checkpoint and metrics.")
    if json.loads(manifest_path.read_text()).get("identity") != identity:
        raise SystemExit("Run/config/source/preflight identity changed; refusing mixed-run continuation.")
    status = json.loads((run / "checkpoints/latest_status.json").read_text())
    checkpoint_stat = (run / "checkpoints/latest.pt").stat()
    if status.get("checkpoint_size") != checkpoint_stat.st_size or status.get("checkpoint_mtime_ns") != checkpoint_stat.st_mtime_ns:
        raise SystemExit("Latest checkpoint status is stale; refusing unsafe automatic continuation.")
    if status.get("training_complete") is True:
        raise SystemExit("Training is already complete; submit final evaluation explicitly.")
    if status.get("resumable") is not True or status.get("replay_saved") is not True:
        raise SystemExit("Continuation requires a resumable checkpoint with persisted replay.")
else:
    if manifest_path.exists() or (run / "metrics.jsonl").exists() or (run / "checkpoints/latest.pt").exists():
        raise SystemExit("Fresh full training refuses an existing run; use its explicit continuation workflow.")
    with manifest_path.open("x") as file:
        json.dump({"identity": identity, "started_at": datetime.now(timezone.utc).isoformat(), "allocation": allocation}, file, indent=2)
print(f"Verified full-suite preflight; GPU={torch.cuda.get_device_name(0)}; chunk={chunk}/{maximum}")
PY

progress_line="$("${PYTHON}" - "${RUN_DIR}" "${CONTINUE_TRAINING}" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
path = run / "metrics.jsonl"
last = {"step": 0, "updates": 0}
if path.exists():
    with path.open() as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row.get("phase") == "off_policy_training":
                    last = row
last_eval_step = -1
if int(sys.argv[2]):
    status = json.loads((run / "checkpoints/latest_status.json").read_text())
    last_eval_step = int(status.get("last_eval_step", -1))
print(int(last["step"]), int(last["updates"]), last_eval_step)
PY
)"
read -r before_step before_updates before_eval_step <<<"${progress_line}"
if ((before_step > 100000000)); then
  echo "Training metrics exceed the immutable 100M budget." >&2
  exit 1
fi
sidecar_job=""
if [[ "${M4PO_ENABLE_WANDB}" == "1" ]]; then
  if sidecar_response="$(sbatch --parsable --partition="${M4PO_PARTITION}" \
    --cpus-per-task=2 --mem=4G --gres=none --time=24:00:00 \
    --chdir="${REPO_ROOT}" --output="${RUN_DIR}/slurm-wandb-%j.out" \
    --export="ALL,TRAINING_JOB_ID=${SLURM_JOB_ID},EVALUATION_JOB_ID=${SLURM_JOB_ID}" \
    "${REPO_ROOT}/scripts/sync_wandb_hpc.sbatch")"; then
    sidecar_job="${sidecar_response%%;*}"
    if [[ ! "${sidecar_job}" =~ ^[0-9]+$ ]]; then
      echo "W&B sidecar returned an unrecognized job ID; training will continue." >&2
      sidecar_job=""
    fi
  else
    echo "W&B sidecar submission failed; local metrics/checkpoints remain authoritative." >&2
  fi
fi
echo "W&B run: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}; sidecar ${sidecar_job:-not submitted}"
args=(
  -m m4po.train --config "${CONFIG}" --seed "${SEED}"
  --total-steps 100000000 --device cuda --log-dir "${RUN_DIR}"
  --save-replay --max-wall-time-seconds "${MAX_WALL_TIME_SECONDS}"
)
if [[ "${CONTINUE_TRAINING}" == "1" ]]; then
  args+=(--resume-checkpoint "${RUN_DIR}/checkpoints/latest.pt")
fi
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
echo "Chunk ${M4PO_CHUNK_INDEX}: starting at ${before_step} transitions, ${before_updates} updates, evaluation step ${before_eval_step}."
"${PYTHON}" "${args[@]}"

progress_line="$("${PYTHON}" - "${RUN_DIR}" "${before_step}" "${before_updates}" "${before_eval_step}" "${CONTINUE_TRAINING}" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
before_step, before_updates, before_eval_step, continuing = map(int, sys.argv[2:])
last = None
with (run / "metrics.jsonl").open() as file:
    for line in file:
        if line.strip():
            row = json.loads(line)
            if row.get("phase") == "off_policy_training":
                last = row
if last is None or not (run / "checkpoints/latest.pt").is_file():
    raise SystemExit("Training did not produce final metrics and a resumable latest checkpoint.")
status = json.loads((run / "checkpoints/latest_status.json").read_text())
checkpoint_stat = (run / "checkpoints/latest.pt").stat()
if status.get("checkpoint_size") != checkpoint_stat.st_size or status.get("checkpoint_mtime_ns") != checkpoint_stat.st_mtime_ns:
    raise SystemExit("Latest checkpoint status is stale; no follow-up job will be submitted.")
if status.get("resumable") is not True or status.get("replay_saved") is not True:
    raise SystemExit("Latest checkpoint is not resumable with its replay buffer.")
step, updates = int(status["step"]), int(status["update_count"])
last_eval_step = int(status.get("last_eval_step", -1))
if (step, updates) != (int(last["step"]), int(last["updates"])):
    raise SystemExit("Final training metrics disagree with latest checkpoint status.")
complete = status.get("training_complete") is True
if (
    step < before_step or updates < before_updates or last_eval_step < before_eval_step
    or (step, updates, last_eval_step) == (before_step, before_updates, before_eval_step)
):
    raise SystemExit("No monotonic training progress; refusing an automatic restart loop.")
if (
    step > 100_000_000 or last_eval_step > step
    or (complete and (step != 100_000_000 or last_eval_step != step or status.get("pending_evaluation") is not False))
    or (not complete and status.get("stop_reason") != "wall_time_limit")
):
    raise SystemExit("Training did not stop at the configured budget or a safe wall-time boundary.")
if continuing and last.get("replay_restored") is not True:
    raise SystemExit("Continuation rebuilt rather than restored replay; refusing to conceal that protocol change.")
print(step, updates, last_eval_step, int(complete))
PY
)"
read -r after_step after_updates after_eval_step training_complete <<<"${progress_line}"
echo "Chunk ${M4PO_CHUNK_INDEX}: committed ${after_step} transitions, ${after_updates} updates, evaluation step ${after_eval_step}."

dependency="afterok:${SLURM_JOB_ID}"
if [[ -n "${sidecar_job}" ]]; then
  dependency+=",afterany:${sidecar_job}"
fi
if [[ "${training_complete}" != "1" ]]; then
  if ((M4PO_CHUNK_INDEX >= MAX_CHUNKS)); then
    echo "Stopped at the ${MAX_CHUNKS}-chunk safety cap; latest checkpoint is preserved. Benchmark is incomplete." >&2
    exit 1
  fi
  next_chunk=$((M4PO_CHUNK_INDEX + 1))
  submitted="$(sbatch --parsable --dependency="${dependency}" \
    --partition="${M4PO_PARTITION}" --chdir="${REPO_ROOT}" \
    --output="${RUN_DIR}/slurm-%x-%j.out" \
    --export="ALL,CONTINUE_TRAINING=1,M4PO_CHUNK_INDEX=${next_chunk}" \
    "${REPO_ROOT}/scripts/train_mmbench_full_hpc.sbatch")"
  stage="continuation"
else
  submitted="$(sbatch --parsable --dependency="${dependency}" \
    --partition="${M4PO_PARTITION}" --chdir="${REPO_ROOT}" \
    --output="${RUN_DIR}/slurm-%x-%j.out" --export=ALL \
    "${REPO_ROOT}/scripts/evaluate_mmbench_hpc.sbatch")"
  stage="final_evaluation"
fi
next_job="${submitted%%;*}"
if [[ ! "${next_job}" =~ ^[0-9]+$ ]]; then
  echo "Unexpected Slurm submission response: ${submitted}" >&2
  exit 1
fi
"${PYTHON}" - "${RUN_DIR}" "${SLURM_JOB_ID}" "${next_job}" "${stage}" \
  "${M4PO_CHUNK_INDEX}" "${after_step}" "${after_updates}" "${after_eval_step}" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

run, current_job, next_job, stage = sys.argv[1:5]
chunk, step, updates, last_eval_step = map(int, sys.argv[5:])
with (Path(run) / "job_chain.jsonl").open("a") as file:
    file.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "job_id": current_job, "next_job_id": next_job, "next_stage": stage, "chunk": chunk, "step": step, "updates": updates, "last_eval_step": last_eval_step}) + "\n")
PY
echo "Submitted ${stage} job ${next_job}, dependent on successful completion of ${SLURM_JOB_ID}."
