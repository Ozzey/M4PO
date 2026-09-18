# MMBench / Newt benchmark

This integration targets the continuous-control benchmark released with
[Newt](https://github.com/nicklashansen/newt), not the unrelated OpenCompass
vision-language MMBench. Use the official environment wrappers and their reward,
action-repeat, timeout, and normalized-score definitions unchanged.

## Source and task selection

The pinned upstream revision is
`1d3fc058b81ddf8d36a5457c29ea407dd374c1b8`. The default checkout is
`external/newt`; `M4PO_NEWT_ROOT` may point to another copy at that revision.
The canonical 200-task training list is `TASK_SET["soup"]` in upstream
`tdmpc2/common/__init__.py`. Do not use every key in `tasks.json`: that metadata
file also contains tasks outside the training split.

The training split contains DMControl (21), DMControl Extended (16), Meta-World
(49), ManiSkill (36), MuJoCo (6), Box2D (8), RoboDesk (6), OGBench (12),
MiniArcade (19), and Atari (27). Upstream metadata supplies instructions,
512-dimensional frozen language embeddings, action dimensions, and task-specific
episode limits. Shared state/action interfaces use padding and action masks;
padding must not create executable action coordinates. The MMBench adapter
preserves in-progress episodes across collection blocks, so a `rollout_steps`
value below a task's native episode length does not discard its transitions.
Only completed episodes enter replay; collection boundaries are not environment
termination or truncation events.

## Pilot versus full comparison

`mmbench_pilot.yaml` is a small, state-only integration and learning pilot.
It is **not a full MMBench result**, and neither an untrained smoke evaluation
nor a reduced-step pilot is comparable to the published Newt aggregate.
`mmbench_all.yaml` selects the full training suite; installing and validating all
ten domain dependencies is required before launching it. This configuration
starts from scratch. `mmbench_pretrained.yaml` instead initializes from a
completed M4PO demonstration-pretraining stage. Neither is a paper-matched Newt
run merely because it uses the same task set and total transition budget.

The released Newt configuration uses 100 million total online transitions,
one worker for each of the 200 tasks, a roughly 20-million-parameter model,
batch size 1,024, and 0.075 optimizer updates per transition. Its default method
also performs 200,000 demonstration-pretraining updates and mixes demonstrations
with online replay. Report M4PO's initialization protocol, model size, task set,
seed, and interaction budget alongside scores. Its online `pretrain_updates`
field means replay-only model updates after random seed collection, not expert
pretraining. The separate `m4po.pretrain_demonstrations` command performs actual
demonstration learning. After that stage, M4PO retains its M3PO-style off-policy
Q-maximizing actor objective: it does not mix demonstrations or add a BC penalty
during online RL. This differs from Newt's continued demo mixing/action
supervision and must be disclosed in the comparison.
See the [official configuration](https://github.com/nicklashansen/newt/blob/1d3fc058b81ddf8d36a5457c29ea407dd374c1b8/tdmpc2/config.py).

The paper reports 11.2 days on one RTX 3090 for its 100-million-step state-based
run, or 4.6 days on two RTX 5090s. Those are **Newt's** timings, not an M4PO
runtime estimate; measure M4PO throughput before committing to a full budget.
See [the paper, Table 3](https://arxiv.org/html/2511.19584v1#S4.T3).

## HPC installation

The HPC scripts refuse to run unless the current hostname belongs to an active
Slurm allocation. Submit jobs from the login node, but run dependency setup,
simulation, training, and evaluation only on allocated compute nodes.

The setup script creates `envs/env_mmbench` as a separate virtual environment.
It reuses the existing IsaacLab environment's CUDA/PyTorch packages through
`--system-site-packages`, while installing benchmark-specific packages only in
the new environment. It never changes the IsaacLab environment. Override
`M4PO_BASE_PYTHON` to use a different existing CUDA-enabled Python installation.
Do not use that base environment for concurrent package upgrades while the
benchmark environment is running.

On an allocated compute node:

```bash
export M4PO_HPC_ROOT=/l/users/${USER}/m4po_hpc
export M4PO_REPO_ROOT=${M4PO_HPC_ROOT}/M4PO
bash scripts/setup_hpc_mmbench.sh pilot
```

The pilot installs Gymnasium 0.29.1, MuJoCo 3.3.6, and DMControl 1.0.34. It uses
NumPy 1.26 or later, below 2.0, to satisfy M4PO's existing requirement, rather
than Newt's original NumPy 1.24.4. The reused Python/PyTorch versions also depend
on the base environment. Record installed versions: this is an environment
adapter for M4PO, not a bit-for-bit reproduction of Newt's software stack.

For all domains, use `bash scripts/setup_hpc_mmbench.sh all`. This additionally
installs the benchmark's Meta-World fork at
`22904d1f65afe920be4325d482808c385f4c0c38`, ManiSkill nightly
`2025.9.19.39`, OGBench 1.1.5, RoboDesk 1.0.0, ALE 0.10.0, and Box2D/Pygame.
The official ManiSkill asset archive is downloaded to a dedicated asset
directory. Full-suite installation can additionally require system OpenGL,
Vulkan, compiler, and rendering libraries; inspect the upstream
[installation definition](https://github.com/nicklashansen/newt/blob/1d3fc058b81ddf8d36a5457c29ea407dd374c1b8/docker/environment.yaml)
and Dockerfile. Validate every domain on the target compute node before a long
run. No tasks should be silently skipped because dependencies are missing.

ManiSkill reads `MS_ASSET_DIR`, pointing at the extracted `.maniskill` directory
(without a `/data` suffix). The launchers set this explicitly; the adapter also
accepts the older `MANISKILL_ASSET_DIR` spelling as a fallback before import.

ALE 0.10.0 includes Atari ROMs in its Python distribution. A separate AutoROM
download or interactive `accept-rom-license` step is not required; this changed
in [ALE 0.9.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.9.0).
The full-suite preflight must still create the native continuous-action Atari
wrappers and verify each task works with the installed Gymnasium version.
Demonstrations are public at
[Hugging Face](https://huggingface.co/datasets/nicklashansen/mmbench), but are not
downloaded or used by either from-scratch configuration.

## Launch and outputs

After setup, a pilot can be submitted with:

```bash
sbatch --export=ALL,CONFIG=m4po/configs/mmbench_pilot.yaml,SEED=0 \
  scripts/train_mmbench_hpc.sbatch
```

Supported overrides include `M4PO_HPC_ROOT`, `M4PO_REPO_ROOT`, `M4PO_PYTHON`,
`M4PO_NEWT_ROOT`, `CONFIG`, `TOTAL_STEPS`, `RUN_DIR`, `SEED`, and
`EVAL_EPISODES`. The launcher deliberately starts fresh and rejects an existing
run directory with metrics/checkpoints or `RESUME_CHECKPOINT`. Old on-policy
IsaacLab checkpoints are not initialization weights for this benchmark.

Training writes metrics and checkpoints into `RUN_DIR`. After successful
training, the same allocation evaluates the frozen final checkpoint and writes
`benchmark.json` and `benchmark.csv`. A Slurm timeout or training exception is
not a successful benchmark completion; inspect job exit status and both outputs.
The launcher defaults to exactly 10 final episodes per task. Newt's released
periodic evaluation specifies at least two episodes per worker but continues
stepping every worker until the longest-horizon tasks finish: with horizons of
25–1,000 steps this collects 2–80 episodes per task. The protocols therefore
have different sampling variance; do not describe 10 per task as uniformly more
evaluation than Newt.

### Full-suite multi-day workflow

The full workflow uses separate preflight, training, and frozen-evaluation jobs.
After installing all dependencies, select a new run directory and validate all
200 native tasks on an allocated node:

```bash
export M4PO_HPC_ROOT=/l/users/${USER}/m4po_hpc
export M4PO_REPO_ROOT=${M4PO_HPC_ROOT}/M4PO
export RUN_DIR=${M4PO_HPC_ROOT}/runs/mmbench_all_seed0
export PREFLIGHT_REPORT=${RUN_DIR}/preflight.json
export CONFIG=m4po/configs/mmbench_all.yaml
export SEED=0
sbatch --export=ALL scripts/preflight_mmbench_hpc.sbatch
```

The preflight runs the test suite and isolated native-task probes, with two
parallel probes and a 180-second timeout per task. Inspect its successful Slurm
exit and `preflight.json`; missing tasks or failures block full training. Next,
the mandatory joint smoke validates simultaneous all-200-task workers, 10,000
real collection steps, nonzero replay updates, and a production-model GPU
profile at action batch 200 and update batch 1,024:

```bash
export JOINT_SMOKE_REPORT=${RUN_DIR}/joint_smoke/smoke_passed.json
sbatch --export=ALL,RUN_DIR=${RUN_DIR}/joint_smoke \
  scripts/smoke_mmbench_hpc.sbatch
```

Inspect the successful job and `smoke_passed.json`, then submit full training:

```bash
sbatch --export=ALL scripts/train_mmbench_full_hpc.sbatch
```

This launcher fixes the budget at 100M transitions, one worker per task (200),
batch size 1,024, and a ten-million-transition replay buffer. It verifies full
native-task coverage and matching source/catalog hashes before starting fresh.
`launch_manifest.json` freezes the configuration, seed, preflight/smoke reports,
native sources, M4PO Python/YAML sources, and workflow scripts across chunks.
Do not edit the deployed release during the campaign. A run-directory lock
prevents overlapping training and final evaluation.

Each training allocation requests 16 CPUs, 96 GB RAM, one GPU, and 24 hours.
Together with the 2-CPU/4-GB logging sidecar, this fits the current account's
24-CPU, 110,000-MB, two-running-job limits on the `ws-ia` partition.
The trainer stops at an optimizer boundary after at most 22 hours by default
(`MAX_WALL_TIME_SECONDS=79200`), leaving time to write a replay-inclusive
`checkpoints/latest.pt`. Successful chunks automatically submit a dependent
continuation, preserving replay, optimizer state, and counters. Simulator state
is not serialized: incomplete episodes at restarts are counted as discarded
partial transitions. This is not uninterrupted simulator-state equivalence.
At most 128 chunks are submitted (`MAX_CHUNKS` may lower the cap). A crash,
failed submission, missing/stale checkpoint-status file, missing replay, or
non-increasing transition/update/evaluation counters stops the chain rather than silently
starting a new experiment. `job_chain.jsonl` records dependent job IDs.

Completion requires 100M collected transitions, all pending optimizer updates,
and the final scheduled learning-curve evaluation. A continuation that only
finishes a pending same-step evaluation counts as valid progress; reaching step
100M alone does not trigger the separate final benchmark. The final
training chunk submits `evaluate_mmbench_hpc.sbatch` as a separate GPU job.
It atomically derives replay-free `checkpoints/final_model.pt` from the verified
complete latest checkpoint, evaluates ten episodes on each of the 200 tasks,
and writes `benchmark.json`, `benchmark.csv`, and `comparison.{json,csv,md}`.
Historical step archives remain unchanged. Training also evaluates two episodes
per task at step zero and every 2M transitions for the comparison learning curve.

W&B synchronization defaults to enabled (`M4PO_ENABLE_WANDB=0` disables it).
Each chunk has a CPU sidecar using one stable run ID; continuation waits for the
preceding sidecar to finish so cursor writers do not overlap. Training sidecars
sync metrics only. Final evaluation's sidecar uploads the same lightweight
`final_model.pt` that was evaluated, not the multi-GB replay checkpoint.
Failure to submit a sidecar is warned about but does not discard local progress.
The default project is `adityanarendra5/m4po-mmbench`.

Online charts use `environment_step` (global transitions across all 200 workers):

- `train/rolling_success_rate`: task-macro native success, using up to the last
  20 completed episodes per contributing task. Tasks without native success are
  excluded, not counted as failures. This is not an all-200 success percentage.
- `train/rolling_success_task_coverage` and `train/rolling_success_task_count`:
  the fraction/count of configured tasks contributing to that success curve.
- `train/rolling_normalized_score` and `train/rolling_score_task_coverage`:
  the corresponding task-macro native score and coverage, including reward-only
  tasks. Full coverage is needed for an all-task interpretation.
- `train/task/<task>/rolling_success_rate` and
  `train/task/<task>/rolling_normalized_score`: per-task curves where defined.

Rolling windows are checkpointed and restored across Slurm chunks. Older
checkpoints lacking these windows explicitly log that history was reset.
Pretraining logs use the separate `pretrain_update` axis and `pretrain/*` loss
series: offline optimizer updates are never counted as environment timesteps.
`scripts/prepare_mmbench_wandb.py` can prepare both W&B run links on an allocated
node without logging dummy success values; the stage sidecars resume those IDs.

The initial full campaign uses **seed 0 only**. Its aggregate is a single-run
M4PO comparison against the released Newt and TD-MPC2 curves, not a five-seed
reproduction. Report whether it used the scratch or demonstration-pretrained
configuration. Repeat with distinct seeds and fresh run directories before
claiming multi-seed uncertainty.

### Demonstration warm start, then the full online benchmark

The supported warm start trains **M4PO's own networks** on official MMBench
demonstrations. Directly loading Newt weights is unsupported: Newt and M4PO have
different encoders, latent structure, conditioning, and prediction heads.

The [official dataset](https://huggingface.co/datasets/nicklashansen/mmbench) is
pinned at `a59d457df617400d3e45a5158c8deac8a52055b4`. All 200 canonical tasks have
shards; together they occupy 23,696,334,612 bytes. On an allocated node, download
only these shards, not the extra task or unrelated assets:

```bash
python -m m4po.download_mmbench_demos \
  --out-dir "${DEMO_DATA_DIR}" --workers 4 --require-slurm
```

The downloader verifies each authoritative LFS SHA-256/size, atomically
publishes completed files, and resumes by rechecking existing files. Its
`manifest.json` uses relative paths and a reproducible digest. Corrupt existing
files are not silently overwritten. The loader verifies the manifest and full
native preflight again, keeps at most the first 20 complete ManiSkill episodes
as in Newt's loader, and retains all other task episodes. It drops unused visual
features and teacher values. Each source episode has T+1 observations and an
initial dummy action/reward row: imported transitions use action/reward/terminal
rows 1 through T. State masks come from verified native observation dimensions;
episode time limits do not become Bellman-terminal failures.

Offline training optimizes M4PO's latent consistency, reward, TD-Q, and
termination losses, while the actor uses masked behavior cloning plus entropy
on detached replay latents. Actor Q maximization is disabled only during this
offline stage; PPO auxiliary updates and explicit exploration bonuses remain
disabled. The production offline budget is **200,000 optimizer updates**, which
are recorded separately from online environment transitions. This is inspired
by Newt's demonstration pretraining, not identical architecture or optimizer
semantics. The online stage uses `seed_steps: 0` and `pretrain_updates: 0`, so it
does not discard the warm start through a new uniform-random seeding phase.

Before the production job, submit the 100-update offline smoke with the
full-sized model/configuration and a real pretrained-to-online weight-loading
check on an allocated GPU:

```bash
export DEMO_SMOKE_RUN_DIR=${M4PO_HPC_ROOT}/runs/mmbench_demo_smoke_seed0
sbatch --export=ALL scripts/smoke_mmbench_demos_hpc.sbatch
```

Keep the completed smoke checkpoint, status file, and `smoke_passed.json`.
After the smoke succeeds, submit the entire
offline-to-online chain from the login node:

```bash
export DEMO_SMOKE_REPORT=${DEMO_SMOKE_RUN_DIR}/checkpoints/latest_status.json
export PRETRAIN_RUN_DIR=${M4PO_HPC_ROOT}/runs/mmbench_demo_pretrain_seed0
export ONLINE_RUN_DIR=${M4PO_HPC_ROOT}/runs/mmbench_demo_online_seed0
export SEED=0
sbatch --export=ALL scripts/pretrain_mmbench_hpc.sbatch
```

`DEMO_DATA_DIR`, `PREFLIGHT_REPORT`, and `JOINT_SMOKE_REPORT` must also be exported.
The pretraining launcher requires matching dataset/config/source hashes,
allocated-GPU smoke evidence, and fresh checkpoint status. It requests the same
16-CPU/96-GB/one-GPU allocation as online training and uses 22-hour safe chunks.
At most eight offline chunks are submitted (`MAX_PRETRAIN_CHUNKS` may lower the
cap). Continuations use the original latest checkpoint, optimizer, replay RNG,
and training RNG state; a failed or non-progressing chunk stops the chain.

After all 200k updates, the immutable `checkpoints/pretrained_model.pt` artifact
initializes a fresh online run through `INIT_CHECKPOINT` and
`mmbench_pretrained.yaml`. The online run hashes that artifact into its launch
identity and retains the same initialization setting on every resume. The
existing full-suite/joint-smoke gates, 100M-transition budget, periodic frozen
evaluations, and final all-task benchmark then apply unchanged. Offline and
online stages have distinct W&B run IDs; offline metrics use optimizer-update
progress rather than being mislabeled as online transitions. No pretrained
artifact is uploaded until the offline stage completes.

## Metrics

The primary benchmark metric is the **unweighted mean across task-level
normalized scores**, using each official wrapper's final `info["score"]`.
It is not an average success percentage. Many tasks have no binary success
criterion; retain missing success as missing instead of treating `NaN` as true
or reporting fabricated zeros as measured failure rates.

Preserve native task-specific episode horizons and score normalization. In
particular, do not apply the earlier IsaacLab Cartpole 128-step survival rule.
Report per-task and per-domain scores together with the aggregate and coverage.
Only call an aggregate a full MMBench score if all 200 training tasks were
evaluated; held-out task adaptation is a separate experiment. The authoritative
aggregation implementation is the upstream
[trainer](https://github.com/nicklashansen/newt/blob/1d3fc058b81ddf8d36a5457c29ea407dd374c1b8/tdmpc2/trainer.py).

## Comparing against released Newt results

Generate a report from a completed, frozen all-200-task evaluation:

```bash
python -m m4po.compare_mmbench runs/mmbench_all_seed0/benchmark.json \
  --curves runs/mmbench_all_seed0/metrics.jsonl \
  --out-prefix runs/mmbench_all_seed0/comparison
```

The command writes JSON, CSV, and Markdown. It refuses pilot coverage, missing
native task scores, inconsistent aggregates, and nonmatching reference budgets.
The official task catalog and reference CSVs are SHA-256 checked against the
pinned Newt revision. Scores are recomputed from all 200 task means, with exact
native domain names for the ten-domain breakdown. Supply multiple benchmark
paths for distinct training seeds and, optionally, one curve path per seed in
the same order. Budgets and final evaluation settings must agree across seeds.

The release's Newt reference is **demonstration-pretrained Newt plus online RL**.
Its average at 100M transitions is `0.4378576328375057`. The separate
`csv/tdmpc2` reference reaches `0.26244842269339813`; it is labeled **released
TD-MPC2**, not silently renamed Newt without demonstrations. The release does
not explicitly identify a separate no-demonstration Newt CSV or supply Newt
per-task CSVs. Those values and per-seed uncertainty remain unavailable in the
report. The main benchmark's number of seeds is not specified by these CSVs;
the paper's explicit five-seed statement concerns held-out adaptation, not the
main 200-task comparison. One M4PO seed is a single-run result, not a
multi-seed reproduction.

Normalized learning-curve AUC is reported only when every supplied seed has
complete all-task frozen evaluations at the official common 2M-step grid from
0 through 100M. It uses trapezoidal integration divided by 100M. Missing step-0
evaluation, partial coverage, duplicate evaluation points, or a shorter budget
produce `null`, not an extrapolated or training-score-based AUC. This requires
saving a genuine pretraining/start-state evaluation at step 0 before learning.
The model's random-data replay warmup is not expert-demonstration pretraining.

M4PO action timing is reported from its measured frozen evaluation, including
device and batch-size metadata. The published Newt CSVs contain no action
latency; the report leaves that reference blank instead of fabricating a
cross-hardware comparison. Model capacity, demonstrations, replay size,
optimizer batch size, task transition exposure, and wall-clock resources must
also be reported when interpreting the score difference. Equal total steps
alone do not make those conditions matched.
