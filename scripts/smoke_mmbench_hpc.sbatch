#!/usr/bin/env bash
#SBATCH --job-name=m4po-mmbench-joint-smoke
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "Refusing to simulate outside a Slurm allocation." >&2
  exit 1
fi

HPC_ROOT="${M4PO_HPC_ROOT:-/l/users/${USER}/m4po_hpc}"
REPO_ROOT="${M4PO_REPO_ROOT:-${HPC_ROOT}/M4PO}"
PYTHON="${M4PO_PYTHON:-${HPC_ROOT}/envs/env_mmbench/bin/python}"
CACHE_ROOT="${M4PO_CACHE_ROOT:-/scratch/${USER}/m4po-cache}"
RUN_DIR="${RUN_DIR:-${HPC_ROOT}/runs/mmbench_joint_smoke_${SLURM_JOB_ID}}"
: "${PREFLIGHT_REPORT:?Set PREFLIGHT_REPORT to the successful full-suite audit.}"
if [[ -e "${RUN_DIR}/metrics.jsonl" || -e "${RUN_DIR}/smoke_passed.json" ]]; then
  echo "Refusing to overwrite an existing smoke run: ${RUN_DIR}" >&2
  exit 1
fi
cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TORCH_DISABLE_DYNAMO=1 TORCHDYNAMO_DISABLE=1 M4PO_DISABLE_TRITON=1
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export M4PO_NEWT_ROOT="${M4PO_NEWT_ROOT:-${REPO_ROOT}/external/newt}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-${MANISKILL_ASSET_DIR:-${HPC_ROOT}/assets/mmbench/.maniskill}}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export TMPDIR="${CACHE_ROOT}/tmp"
mkdir -p "${RUN_DIR}" "${TMPDIR}"
"${PYTHON}" - "${PREFLIGHT_REPORT}" <<'PY'
import json
import sys
from m4po.mmbench_preflight import verify_slurm_allocation

print(verify_slurm_allocation(), flush=True)
with open(sys.argv[1]) as stream:
    report = json.load(stream)
if report.get("full_suite_passed") is not True or report.get("passed_task_count") != 200:
    raise SystemExit("All 200 tasks must pass individual preflight before joint smoke.")
PY

"${PYTHON}" -m m4po.train --config m4po/configs/mmbench_all_smoke.yaml \
  --log-dir "${RUN_DIR}" --device cuda
"${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
from contextlib import closing
from pathlib import Path
import resource
import statistics
import sys
import time
import numpy as np
import torch
from m4po.common.buffer import Episode, EpisodeReplayBuffer, ObservationSpec
from m4po.common.config import M4POConfig
from m4po.envs.mmbench_env import load_mmbench_catalog
from m4po.envs import make_vector_env
from m4po.m4po import M4POAgent
from m4po.mmbench_preflight import verify_slurm_allocation

run = Path(sys.argv[1])
payload = torch.load(run / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
extra = payload["extra"]
if payload["step"] != 10000 or not extra.get("training_complete") or extra["update_count"] <= 0:
    raise SystemExit("Joint smoke did not complete real replay updates.")
cfg = M4POConfig.from_yaml("m4po/configs/mmbench_all.yaml")
metadata, _, _ = load_mmbench_catalog(cfg.mmbench_root)
contexts = np.asarray([metadata[name]["text_embedding"] for name in cfg.task_names], dtype=np.float32)
with closing(make_vector_env(cfg)) as env:
    observation = env.reset()
    if env.num_envs != 200 or env.num_tasks != 200 or set(env.task_ids.tolist()) != set(range(200)):
        raise SystemExit("Production profile requires all 200 distinct native tasks concurrently.")
    if env.observation_spec != ObservationSpec((0, 0, 0), 0, 128) or env.action_dim != 16:
        raise SystemExit("Production native observation/action contract differs from the benchmark.")
    agent = M4POAgent(env.observation_spec, env.action_dim, cfg, torch.device("cuda"), task_contexts=contexts)
    replay = EpisodeReplayBuffer(cfg.replay_capacity, cfg.model_horizon, cfg.batch_size, device="cuda", observation_spec=agent.observation_spec)
    for episode in extra["replay_state"]["episodes"]:
        replay.add(Episode(**episode))
    if not replay.can_sample:
        raise SystemExit("Joint smoke has no completed native episodes.")
    replay_task_count = len({int(episode.task_ids[0]) for episode in replay._episodes})
    torch.cuda.reset_peak_memory_stats()
    action_ms, native_step_ms, update_ms = [], [], []
    device_used_samples = []
    for index in range(6):
        if set(env.task_ids.tolist()) != set(range(200)):
            raise SystemExit("Production profile lost a native task during an auto-reset.")
        kwargs = {
            "task_ids": env.task_ids.copy(),
            "embodiment_ids": env.embodiment_ids.copy(),
            "action_mask": env.action_masks.copy(),
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = agent.act_np(observation, **kwargs)
        torch.cuda.synchronize()
        action_elapsed = (time.perf_counter() - started) * 1000
        if (
            output.action.shape != (200, 16)
            or not np.isfinite(output.action).all()
            or np.any(np.abs(output.action) > 1.0 + 1e-6)
            or np.any(np.abs(output.action * (1.0 - kwargs["action_mask"])) > 1e-7)
        ):
            raise SystemExit("Production planner returned invalid normalized/masked actions.")
        started = time.perf_counter()
        observation, rewards, dones, infos = env.step(output.action)
        torch.cuda.synchronize()
        native_elapsed = (time.perf_counter() - started) * 1000
        env.observation_spec.validate(observation)
        if (
            not all(np.isfinite(value).all() for value in observation.values())
            or np.asarray(rewards).shape != (200,)
            or not np.isfinite(rewards).all()
            or np.asarray(dones).shape != (200,)
            or len(infos) != 200
        ):
            raise SystemExit("Production native step returned an invalid batch.")
        started = time.perf_counter()
        metrics = agent.update(replay)
        torch.cuda.synchronize()
        update_elapsed = (time.perf_counter() - started) * 1000
        if not all(np.isfinite(value) for value in metrics.values()):
            raise SystemExit("Production-sized learner produced non-finite smoke metrics.")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        device_used_samples.append(total_bytes - free_bytes)
        if index:
            action_ms.append(action_elapsed)
            native_step_ms.append(native_elapsed)
            update_ms.append(update_elapsed)
    result = {
        "purpose": "joint 200-task simulator smoke and concurrent production-model compute profile; not a benchmark score",
        "allocation": verify_slurm_allocation(),
        "joint_smoke_steps": payload["step"],
        "joint_smoke_updates": extra["update_count"],
        "tasks": 200,
        "passed": True,
        "production_trainable_parameters": sum(parameter.numel() for parameter in agent.parameters() if parameter.requires_grad),
        "production_action_batch_size": 200,
        "production_action_batch_mean_ms": statistics.mean(action_ms),
        "production_distinct_native_tasks": 200,
        "production_native_batch_step_mean_ms": statistics.mean(native_step_ms),
        "production_update_batch_size": cfg.batch_size,
        "production_update_mean_ms": statistics.mean(update_ms),
        "production_update_replay_task_count": replay_task_count,
        "production_replay_transitions_profiled": len(replay),
        "production_update_data": "completed short-horizon episodes from the 50-vector-step smoke; not all-task replay coverage",
        "production_process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "production_process_memory_definition": "Linux process peak RSS with concurrent native environments and the smoke replay, not a full 10M-transition replay or resume snapshot",
        "production_peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "production_peak_cuda_definition": "PyTorch allocator peak with all 200 native environments alive; excludes external native allocations",
        "production_device_used_bytes_max_sampled": max(device_used_samples),
        "production_device_total_bytes": total_bytes,
        "production_device_memory_definition": "device-wide CUDA total-minus-free sampled after each update, including other processes; not a process-isolated peak",
    }
(run / "smoke_passed.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
