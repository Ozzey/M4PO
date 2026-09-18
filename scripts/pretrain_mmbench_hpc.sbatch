#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench-pretrain
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
  echo "Refusing to pretrain outside a Slurm allocation." >&2
  exit 1
fi
current_host="$(hostname -s)"
matched=0
while IFS= read -r allocated_host; do
  if [[ "${allocated_host%%.*}" == "${current_host}" ]]; then matched=1; break; fi
done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((matched == 0)); then
  echo "Refusing to pretrain on ${current_host}: not an allocated node." >&2
  exit 1
fi

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
: "${PRETRAIN_RUN_DIR:?Set PRETRAIN_RUN_DIR to a fresh offline-stage directory.}"
ONLINE_RUN_DIR="${ONLINE_RUN_DIR:-${RUN_DIR:-}}"
: "${ONLINE_RUN_DIR:?Set ONLINE_RUN_DIR or RUN_DIR to the future online-stage directory.}"
: "${DEMO_DATA_DIR:?Set DEMO_DATA_DIR to the verified official demonstration dataset.}"
: "${PREFLIGHT_REPORT:?Set PREFLIGHT_REPORT to the full native-suite audit.}"
: "${JOINT_SMOKE_REPORT:?Set JOINT_SMOKE_REPORT to the passed joint all-task smoke.}"
: "${DEMO_SMOKE_REPORT:?Set DEMO_SMOKE_REPORT to the completed 100-update offline smoke latest_status.json.}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-m4po/configs/mmbench_all.yaml}"
ONLINE_CONFIG="${ONLINE_CONFIG:-m4po/configs/mmbench_pretrained.yaml}"
SEED="${SEED:-0}"
CONTINUE_PRETRAINING="${CONTINUE_PRETRAINING:-0}"
PRETRAIN_CHUNK_INDEX="${PRETRAIN_CHUNK_INDEX:-1}"
MAX_PRETRAIN_CHUNKS="${MAX_PRETRAIN_CHUNKS:-8}"
MAX_WALL_TIME_SECONDS="${MAX_WALL_TIME_SECONDS:-79200}"
M4PO_PARTITION="${M4PO_PARTITION:-${SLURM_JOB_PARTITION:-ws-ia}}"
if [[ "${PRETRAIN_UPDATES:-200000}" != "200000" || -n "${RESUME_CHECKPOINT:-}" ]]; then
  echo "Production pretraining is fixed at 200k updates; continuation uses its own latest checkpoint." >&2
  exit 1
fi
if [[ "${CONTINUE_PRETRAINING}" != "0" && "${CONTINUE_PRETRAINING}" != "1" ]]; then
  echo "CONTINUE_PRETRAINING must be 0 or 1." >&2
  exit 1
fi
# Sidecars must follow the offline directory, not the future online run.
RUN_DIR="${PRETRAIN_RUN_DIR}"
mkdir -p "${RUN_DIR}" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/xdg"
exec 9>"${RUN_DIR}/.workflow.lock"
if ! flock -n 9; then
  echo "Another process owns offline run ${RUN_DIR}." >&2
  exit 1
fi
cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TORCH_DISABLE_DYNAMO=1 TORCHDYNAMO_DISABLE=1 M4PO_DISABLE_TRITON=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-^openib}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export TMPDIR="${CACHE_ROOT}/tmp" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export M4PO_NEWT_ROOT="${M4PO_NEWT_ROOT:-${REPO_ROOT}/external/newt}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-${MANISKILL_ASSET_DIR:-${HPC_ROOT}/assets/mmbench/.maniskill}}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export M4PO_HPC_ROOT="${HPC_ROOT}" M4PO_REPO_ROOT="${REPO_ROOT}" M4PO_PYTHON="${PYTHON}"
export RUN_DIR PRETRAIN_RUN_DIR ONLINE_RUN_DIR DEMO_DATA_DIR PREFLIGHT_REPORT JOINT_SMOKE_REPORT DEMO_SMOKE_REPORT
export PRETRAIN_CONFIG ONLINE_CONFIG SEED MAX_PRETRAIN_CHUNKS MAX_WALL_TIME_SECONDS M4PO_PARTITION
export M4PO_ENABLE_WANDB="${M4PO_ENABLE_WANDB:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-adityanarendra5}"
export WANDB_PROJECT="${WANDB_PROJECT:-m4po-mmbench}"
export PRETRAIN_WANDB_RUN_ID="${PRETRAIN_WANDB_RUN_ID:-m4po-mmbench-pretrain-seed${SEED}-$(basename "${PRETRAIN_RUN_DIR}")}"
export ONLINE_WANDB_RUN_ID="${ONLINE_WANDB_RUN_ID:-m4po-mmbench-demo-pretrained-seed${SEED}-$(basename "${ONLINE_RUN_DIR}")}"
export WANDB_RUN_ID="${PRETRAIN_WANDB_RUN_ID}"
export WANDB_RUN_NAME="M4PO MMBench demonstration pretraining - seed ${SEED}"
export WANDB_GROUP="mmbench-demonstration-pretraining"
export M4PO_WANDB_CHECKPOINT="${RUN_DIR}/checkpoints/pretrained_model.pt"

before_updates="$("${PYTHON}" - "${CONTINUE_PRETRAINING}" "${PRETRAIN_CHUNK_INDEX}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from m4po.common.config import M4POConfig
from m4po.common.demonstrations import DATASET_MANIFEST_SHA256
from m4po.common.initialization import file_sha256
from m4po.download_mmbench_demos import manifest_digest
from m4po.mmbench_preflight import verify_slurm_allocation
from m4po.pretrain_demonstrations import STAGE, _config_digest, _implementation_digest

continuing, chunk = map(int, sys.argv[1:])
maximum = int(os.environ["MAX_PRETRAIN_CHUNKS"])
wall_time = int(os.environ["MAX_WALL_TIME_SECONDS"])
seed = int(os.environ["SEED"])
if seed < 0 or not 1 <= chunk <= maximum <= 8 or not 60 <= wall_time <= 79200:
    raise SystemExit("Require nonnegative seed, 1 <= chunk <= max_pretrain_chunks <= 8, and 60..79200 seconds per chunk.")
if not continuing and chunk != 1:
    raise SystemExit("Fresh pretraining must begin at chunk 1.")
allocation = verify_slurm_allocation()
if not torch.cuda.is_available():
    raise SystemExit("Offline pretraining requires an allocated visible CUDA device.")
run = Path(os.environ["PRETRAIN_RUN_DIR"])
online_run = Path(os.environ["ONLINE_RUN_DIR"])
if run.resolve() == online_run.resolve():
    raise SystemExit("Offline and online stages need separate run directories.")
config_path = Path(os.environ["PRETRAIN_CONFIG"])
cfg = M4POConfig.from_yaml(config_path)
cfg.seed = seed
cfg.validate()
if (
    cfg.env != "mmbench" or cfg.learning_mode != "off_policy" or cfg.num_tasks != 200
    or cfg.batch_size != 1024 or cfg.init_checkpoint or cfg.resume_checkpoint
):
    raise SystemExit("Offline pretraining requires the full M4PO all-task configuration with batch 1024 and no initialization/resume override.")
dataset = json.loads((Path(os.environ["DEMO_DATA_DIR"]) / "manifest.json").read_text())
dataset_hash = dataset.get("dataset_manifest_sha256")
if dataset_hash != DATASET_MANIFEST_SHA256 or manifest_digest(dataset) != dataset_hash:
    raise SystemExit("Demonstration dataset manifest does not match the pinned full release.")
source_hash = _implementation_digest()
config_hash = _config_digest(cfg)
smoke_path = Path(os.environ["DEMO_SMOKE_REPORT"])
smoke = json.loads(smoke_path.read_text())
smoke_stat = (smoke_path.parent / "latest.pt").stat()
if (
    smoke.get("stage") != STAGE or smoke.get("step") != 0
    or smoke.get("updates_completed") != 100 or smoke.get("target_updates") != 100
    or smoke.get("training_complete") is not True or smoke.get("resumable") is not True
    or smoke.get("dataset_sha256") != dataset_hash
    or smoke.get("m4po_sources_sha256") != source_hash or smoke.get("config_sha256") != config_hash
    or smoke.get("allocation", {}).get("verified") is not True
    or not str(smoke.get("device", "")).startswith("cuda")
    or smoke.get("checkpoint_size") != smoke_stat.st_size
    or smoke.get("checkpoint_mtime_ns") != smoke_stat.st_mtime_ns
):
    raise SystemExit("Require a completed, source/config/dataset-matched, allocated-GPU 100-update offline smoke with fresh checkpoint status.")
smoke_passed_path = smoke_path.parent.parent / "smoke_passed.json"
smoke_passed = json.loads(smoke_passed_path.read_text())
smoke_demo = smoke_passed.get("demonstration_pretraining", {})
if (
    smoke_passed.get("passed") is not True
    or smoke_passed.get("allocation", {}).get("verified") is not True
    or smoke_demo.get("stage_completed") is not True
    or smoke_demo.get("completed_updates") != 100 or smoke_demo.get("task_count") != 200
    or smoke_demo.get("dataset_sha256") != dataset_hash
    or smoke_demo.get("m4po_sources_sha256") != source_hash
    or smoke_demo.get("config_sha256") != config_hash
    or smoke_demo.get("protocol_sha256") != smoke.get("protocol_sha256")
    or smoke_passed.get("online_seed_steps") != 0 or smoke_passed.get("online_replay_warmup_updates") != 0
    or smoke_passed.get("initialization", {}).get("checkpoint_sha256") != file_sha256(smoke_path.parent / "pretrained_model.pt")
):
    raise SystemExit("Offline smoke must also pass the real pretrained-to-online weight-loading check.")
joint_smoke = json.loads(Path(os.environ["JOINT_SMOKE_REPORT"]).read_text())
if joint_smoke.get("passed") is not True or joint_smoke.get("tasks") != 200:
    raise SystemExit("Joint all-task smoke must pass before the offline-to-online campaign.")
control_hash = hashlib.sha256()
for name in ("pretrain_mmbench_hpc.sbatch", "train_mmbench_full_hpc.sbatch", "evaluate_mmbench_hpc.sbatch", "preflight_mmbench_hpc.sbatch", "smoke_mmbench_hpc.sbatch", "smoke_mmbench_demos_hpc.sbatch", "sync_wandb_hpc.sbatch"):
    control_hash.update(name.encode())
    control_hash.update((Path("scripts") / name).read_bytes())
identity = {
    "m4po_sources_sha256": source_hash, "workflow_sha256": control_hash.hexdigest(),
    "config_sha256": config_hash, "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "online_config_sha256": hashlib.sha256(Path(os.environ["ONLINE_CONFIG"]).read_bytes()).hexdigest(),
    "dataset_sha256": dataset_hash, "seed": seed, "target_updates": 200_000,
    "demo_smoke_passed_sha256": hashlib.sha256(smoke_passed_path.read_bytes()).hexdigest(),
    "online_run_dir": str(online_run.resolve()),
    **{name.lower() + "_sha256": hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
       for name in ("PREFLIGHT_REPORT", "JOINT_SMOKE_REPORT", "DEMO_SMOKE_REPORT")},
}
manifest_path = run / "launch_manifest.json"
latest = run / "checkpoints/latest.pt"
before = 0
if continuing:
    if json.loads(manifest_path.read_text()).get("identity") != identity:
        raise SystemExit("Offline run identity changed; refusing mixed-protocol continuation.")
    status = json.loads((run / "checkpoints/latest_status.json").read_text())
    stat = latest.stat()
    if (
        status.get("checkpoint_size") != stat.st_size or status.get("checkpoint_mtime_ns") != stat.st_mtime_ns
        or status.get("stage") != STAGE or status.get("target_updates") != 200_000
        or status.get("resumable") is not True or status.get("training_complete") is not False
        or status.get("dataset_sha256") != dataset_hash
        or status.get("m4po_sources_sha256") != source_hash or status.get("config_sha256") != config_hash
    ):
        raise SystemExit("Offline continuation requires a fresh, matching, incomplete resumable checkpoint status.")
    before = int(status["updates_completed"])
else:
    if manifest_path.exists() or latest.exists() or (run / "metrics.jsonl").exists() or (run / "checkpoints/pretrained_model.pt").exists():
        raise SystemExit("Fresh offline training refuses existing progress; use continuation or a new directory.")
    if (online_run / "launch_manifest.json").exists() or (online_run / "metrics.jsonl").exists() or (online_run / "checkpoints/latest.pt").exists():
        raise SystemExit("Future online run already contains progress; choose a new online directory.")
    with manifest_path.open("x") as file:
        json.dump({"identity": identity, "allocation": allocation, "started_at": datetime.now(timezone.utc).isoformat()}, file, indent=2)
print(before)
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
    if [[ ! "${sidecar_job}" =~ ^[0-9]+$ ]]; then
      echo "Unrecognized W&B sidecar job ID; offline training continues with local metrics." >&2
      sidecar_job=""
    fi
  else
    echo "W&B sidecar submission failed; offline metrics remain available locally." >&2
  fi
fi
args=(
  -m m4po.pretrain_demonstrations --config "${PRETRAIN_CONFIG}"
  --data-dir "${DEMO_DATA_DIR}" --preflight-report "${PREFLIGHT_REPORT}"
  --updates 200000 --log-dir "${RUN_DIR}" --device cuda --seed "${SEED}"
  --max-wall-time-seconds "${MAX_WALL_TIME_SECONDS}" --require-slurm
)
if [[ "${CONTINUE_PRETRAINING}" == "1" ]]; then
  args+=(--resume "${RUN_DIR}/checkpoints/latest.pt")
fi
echo "Offline chunk ${PRETRAIN_CHUNK_INDEX}/${MAX_PRETRAIN_CHUNKS}: starting at ${before_updates}/200000 updates."
echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}"
"${PYTHON}" "${args[@]}"

progress_line="$("${PYTHON}" - "${RUN_DIR}" "${before_updates}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run, before = Path(sys.argv[1]), int(sys.argv[2])
status = json.loads((run / "checkpoints/latest_status.json").read_text())
identity = json.loads((run / "launch_manifest.json").read_text())["identity"]
protocol = json.loads((run / "demonstration_protocol.json").read_text())
latest = run / "checkpoints/latest.pt"
stat = latest.stat()
updates = status.get("updates_completed")
complete = status.get("training_complete") is True
if (
    status.get("checkpoint_size") != stat.st_size or status.get("checkpoint_mtime_ns") != stat.st_mtime_ns
    or status.get("stage") != "demonstration_pretraining" or status.get("step") != 0
    or status.get("resumable") is not True or status.get("target_updates") != 200_000
    or type(updates) is not int or not before < updates <= 200_000
    or type(status.get("training_complete")) is not bool
    or complete != (updates == 200_000)
    or status.get("dataset_sha256") != identity["dataset_sha256"]
    or status.get("m4po_sources_sha256") != identity["m4po_sources_sha256"]
    or status.get("config_sha256") != identity["config_sha256"]
    or status.get("protocol_sha256") != protocol.get("protocol_sha256")
    or (not complete and status.get("stop_reason") != "wall_time_limit")
):
    raise SystemExit("Offline chunk lacks verified monotonic, protocol-matched progress at a safe boundary; not requeuing.")
last = None
with (run / "metrics.jsonl").open() as file:
    for line in file:
        if line.strip():
            row = json.loads(line)
            if row.get("phase") == "demonstration_pretraining":
                last = row
if last is None or last.get("pretrain_updates") != updates or last.get("training_complete") is not complete:
    raise SystemExit("Offline checkpoint status and final metrics disagree.")
if complete:
    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as file:
            for data in iter(lambda: file.read(8 * 1024 * 1024), b""):
                result.update(data)
        return result.hexdigest()
    if digest(run / "checkpoints/pretrained_model.pt") != digest(latest):
        raise SystemExit("Published pretrained artifact does not match the completed latest checkpoint.")
print(updates, int(complete))
PY
)"
read -r after_updates pretraining_complete <<<"${progress_line}"
dependency="afterok:${SLURM_JOB_ID}"
if [[ -n "${sidecar_job}" ]]; then dependency+=",afterany:${sidecar_job}"; fi
if [[ "${pretraining_complete}" == "1" ]]; then
  submitted="$(sbatch --parsable --dependency="${dependency}" \
    --partition="${M4PO_PARTITION}" --chdir="${REPO_ROOT}" \
    --output="${PRETRAIN_RUN_DIR}/slurm-online-%j.out" \
    --export="ALL,CONFIG=${ONLINE_CONFIG},RUN_DIR=${ONLINE_RUN_DIR},INIT_CHECKPOINT=${PRETRAIN_RUN_DIR}/checkpoints/pretrained_model.pt,CONTINUE_TRAINING=0,M4PO_CHUNK_INDEX=1,WANDB_RUN_ID=${ONLINE_WANDB_RUN_ID},WANDB_RUN_NAME=M4PO MMBench all 200 demo-pretrained - seed ${SEED},WANDB_GROUP=mmbench-all-demo-pretrained" \
    "${REPO_ROOT}/scripts/train_mmbench_full_hpc.sbatch")"
  stage="online_training"
else
  if ((PRETRAIN_CHUNK_INDEX >= MAX_PRETRAIN_CHUNKS)); then
    echo "Offline ${MAX_PRETRAIN_CHUNKS}-chunk safety cap reached; checkpoint preserved, online stage not launched." >&2
    exit 1
  fi
  next_chunk=$((PRETRAIN_CHUNK_INDEX + 1))
  submitted="$(sbatch --parsable --dependency="${dependency}" \
    --partition="${M4PO_PARTITION}" --chdir="${REPO_ROOT}" \
    --output="${RUN_DIR}/slurm-%x-%j.out" \
    --export="ALL,CONTINUE_PRETRAINING=1,PRETRAIN_CHUNK_INDEX=${next_chunk}" \
    "${REPO_ROOT}/scripts/pretrain_mmbench_hpc.sbatch")"
  stage="offline_continuation"
fi
next_job="${submitted%%;*}"
if [[ ! "${next_job}" =~ ^[0-9]+$ ]]; then
  echo "Unexpected Slurm submission response: ${submitted}" >&2
  exit 1
fi
"${PYTHON}" - "${RUN_DIR}" "${SLURM_JOB_ID}" "${next_job}" "${stage}" "${PRETRAIN_CHUNK_INDEX}" "${after_updates}" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

run, current_job, next_job, stage = sys.argv[1:5]
chunk, updates = map(int, sys.argv[5:])
with (Path(run) / "job_chain.jsonl").open("a") as file:
    file.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "job_id": current_job, "next_job_id": next_job, "next_stage": stage, "chunk": chunk, "pretrain_updates": updates}) + "\n")
PY
echo "Committed ${after_updates}/200000 offline updates; submitted ${stage} job ${next_job}."
