# Development and usage

Operational guidance retained from the previous project README. Commands are
relative to the repository root. These examples describe the implementation,
not a reproduction of the manuscript's reported experiments.

## Getting started

Use Python 3.10–3.13.

```bash
bash setup_venv.sh
source .venv/bin/activate
```

Run the complete simulator-free smoke path:

```bash
bash scripts/smoke_test_mock.sh
```

Or invoke the CLIs directly:

```bash
python -m m4po.train \
  --config m4po/configs/mock_smoke.yaml \
  --log-dir artifacts/runs/mock_smoke

python -m m4po.evaluate \
  --checkpoint artifacts/runs/mock_smoke/checkpoints/latest.pt \
  --episodes 2 \
  --device cpu \
  --out artifacts/runs/mock_smoke/benchmark.json
```

Resume from a compatible completed-boundary checkpoint:

```bash
python -m m4po.train \
  --config m4po/configs/mock_smoke.yaml \
  --resume-checkpoint artifacts/runs/mock_smoke/checkpoints/latest.pt \
  --total-steps 128 \
  --log-dir artifacts/runs/mock_smoke_resumed
```

Off-policy checkpoints save model, Q targets, optimizers, and training state.
By default, replay is rebuilt after resume with random warmup; completed
pretraining is not repeated. With `save_replay: true`, `latest.pt` restores
completed episodes and sampling state without another warmup. Unfinished
simulator episodes still restart. Schema-2 checkpoints are interpreted as legacy `on_policy` checkpoints;
they cannot resume into the off-policy learner. Start a new run to change
learning modes.

Standalone benchmarking defaults to the paper's 100 episodes per
task–embodiment pair. The shorter command above is intended only as a smoke
test; reported experiments should also aggregate independently trained seeds.

## NEWT / MMBench

Initialize the pinned benchmark source with `git submodule update --init external/newt`.
See [`docs/MMBENCH.md`](MMBENCH.md) for HPC setup and protocol details.
The four-task [`mmbench_pilot.yaml`](../m4po/configs/mmbench_pilot.yaml) validates
joint off-policy learning before scaling to the 200-task configuration. Both
start from scratch without expert demonstrations. The separate
`m4po.pretrain_demonstrations` stage trains M4PO on the pinned official dataset;
`mmbench_pretrained.yaml` then initializes from its completed M4PO checkpoint.
Direct NEWT weight transfer is not supported because the architectures differ.
This demo-pretrained variant still differs from NEWT: it does not mix expert
data into online replay. Report MMBench's native task-averaged
normalized **score**, not a fabricated success percentage for reward-only tasks.

## IsaacLab Forge multi-task training

IsaacLab 2.3.0 and Isaac Sim 5.1.0 are installed on this host in the
`env_isaaclab` Conda environment. Use that interpreter, not the repository's
CPU virtual environment:

```bash
export ISAACLAB_PYTHON=/home/aditya/miniconda3/envs/env_isaaclab/bin/python
```

Run the short end-to-end train/checkpoint/evaluate smoke test:

```bash
bash scripts/smoke_test_isaaclab_forge.sh
```

Run the longer three-task configuration and then evaluate it:

```bash
"${ISAACLAB_PYTHON}" -m m4po.train \
  --config m4po/configs/isaaclab_forge.yaml

"${ISAACLAB_PYTHON}" -m m4po.evaluate \
  --checkpoint artifacts/runs/isaaclab_forge/checkpoints/latest.pt \
  --episodes 100 \
  --device cuda \
  --out artifacts/runs/isaaclab_forge/benchmark.json \
  --csv-out artifacts/runs/isaaclab_forge/benchmark.csv
```

Each stock Forge task runs in an isolated simulator child process with one
global SimulationContext. Forge configs keep `eval_every: 0`: training must
release and reap its simulator child before standalone evaluation starts, so a
second Isaac Sim worker does not contend for GPU memory. The smoke script
already enforces this lifecycle. Do not enable in-process periodic evaluation
for this adapter.

During training, all vector workers run one task for a collection block.
At the next collection boundary, a seeded shuffled deck selects a task; when the
selection changes, the adapter closes and reaps the current simulator child
before spawning one for the new task. Every three blocks therefore cover all
three Forge tasks once in shuffled order. This is balanced sequential
multi-task collection. Completed episodes from previous tasks remain in replay
and can be sampled while another task is active.

Evaluation uses the adapter's `sequential_evaluation` protocol. It selects one
task–embodiment pair, evaluates all workers on that pair, closes and reaps that
simulator child, and proceeds to the next pair in a fresh child. The common
evaluator still reports per-pair, per-task, aggregate, and worst-embodiment
statistics.

The built-in adapter currently supports deployable policy-vector observations
and continuous vector actions for one common embodiment; cameras and privileged
critic observations are excluded. Tasks must share simulation `dt` and control
decimation, and switching tasks incurs simulator reconstruction overhead.
Forge assets, an NVIDIA GPU, and a working IsaacLab installation are required.

The supplied IsaacLab configurations use `updates_per_step: 0.03125`, meaning
one replay gradient update per vector step with 32 workers. This deliberately
reduces compute relative to the M3PO-style default of one update per individual
environment transition (`1.0`). They use a replay sequence batch size of 256,
1,000 random seed transitions, and 1,000 initial pretraining updates after
replay becomes sampleable. The smoke configs shorten these budgets to exercise
collection, replay, optimization, checkpointing, and evaluation quickly; they
are not policy-quality benchmarks. All supplied configs disable the explicit
exploration bonus and PPO-like auxiliary.

Artifacts mirror M3PO:

```text
<log_dir>/
├── config_resolved.yaml
├── metrics.jsonl
└── checkpoints/
    ├── step_<N>.pt
    ├── latest.pt
    ├── interrupted.pt   # when interrupted
    └── emergency.pt     # when an update fails
```

## IsaacLab Cartpole multi-task training

For a faster learning run, the Cartpole configuration alternates between the
prebuilt `Isaac-Cartpole-Direct-v0` and `Isaac-Cartpole-v0` registrations.
They expose the same continuous four-dimensional observation and
one-dimensional action interface, but exercise IsaacLab's direct and
manager-based environment implementations. Collection remains balanced at
collection boundaries. The adapter aligns their timeout counting and adds the
Direct task's pole-angle failure bound to the manager-based configuration so
episode success has the same meaning for both registrations.

Run local train-and-frozen-evaluation coverage with:

```bash
bash scripts/smoke_test_isaaclab_cartpole.sh
```

On the HPC, submit the smoke test first and the full run only after it passes:

```bash
sbatch --export=ALL,OMNI_KIT_ACCEPT_EULA=YES \
  scripts/smoke_isaaclab_cartpole_hpc.sbatch

sbatch --export=ALL,OMNI_KIT_ACCEPT_EULA=YES \
  scripts/train_isaaclab_cartpole_hpc.sbatch
```

Both batch scripts refuse to run outside their Slurm allocation. The full
configuration performs 64 collection blocks, evenly split between the two tasks, and
the smoke path trains each task twice before evaluating the frozen checkpoint
on both registrations. The installed IsaacLab 2.3.0 registry and its 152 task
IDs are recorded in
[`docs/ISAACLAB_TASK_REGISTRY.md`](ISAACLAB_TASK_REGISTRY.md).

## Frozen task-language contexts

The paper uses a frozen CLIP text encoder. Precompute task vectors once and
point `task_context_path` at the resulting `.npy` file:

```bash
python scripts/encode_task_contexts.py \
  --texts "reach the red block" "push the blue cube" \
  --out artifacts/task_contexts.npy

python -m m4po.train \
  --config m4po/config.yaml \
  --task-context-path artifacts/task_contexts.npy
```

This command requires the optional dependency and locally available model
weights:

```bash
python -m pip install -e '.[clip]'
```

Set `task_context_mode: learned` for the paper's learned task-ID ablation, or
`task_context_mode: none` to use one task-independent context. The built-in
mock environment supplies deterministic frozen context vectors, so the default
smoke path exercises the frozen-context interface without downloading weights.

## Code map

| Concept | Code |
|---|---|
| Agent, off-policy Q/actor/world-model updates, checkpoints | [`m4po/m4po.py`](../m4po/m4po.py) |
| Multimodal hierarchical world model and EMA targets | [`m4po/common/world_model.py`](../m4po/common/world_model.py) |
| Fixed-noise Gaussian actor | [`m4po/common/policy.py`](../m4po/common/policy.py) |
| Stochastic differentiable MPPI | [`m4po/common/planner.py`](../m4po/common/planner.py) |
| Episodic replay and legacy fresh rollout storage | [`m4po/common/buffer.py`](../m4po/common/buffer.py) |
| Action normalization/tokenization | [`m4po/common/action_tokenizer.py`](../m4po/common/action_tokenizer.py) |
| GAE, lambda returns, masks, discrepancy | [`m4po/common/losses.py`](../m4po/common/losses.py) |
| Online collection/update loop | [`m4po/trainer/online_trainer.py`](../m4po/trainer/online_trainer.py) |
| Mock multi-task/multi-embodiment environment | [`m4po/envs/mock_env.py`](../m4po/envs/mock_env.py) |
| Lazy IsaacLab factory loader | [`m4po/envs/isaaclab_env.py`](../m4po/envs/isaaclab_env.py) |
| Sequential prebuilt IsaacLab adapter | [`m4po/envs/isaaclab_prebuilt.py`](../m4po/envs/isaaclab_prebuilt.py) |
| Aggregate and worst-embodiment evaluation | [`m4po/common/evaluation.py`](../m4po/common/evaluation.py) |
| Flat YAML/CLI configuration | [`m4po/common/config.py`](../m4po/common/config.py) |

## Algorithm order

With the default `learning_mode: off_policy`, the trainer:

1. collects masked uniform random actions during warmup, then stochastic MPPI
   actions;
2. adds completed episodes to bounded replay, preserving true final
   observations and each episode's task and embodiment;
3. samples fixed-horizon sequences from replay and updates the hierarchical
   world model and Q ensemble with extrinsic rewards;
4. maximizes Q for actor actions at detached replay latents, with entropy
   coefficient `1e-4` and Q parameters frozen during actor optimization;
5. updates the EMA targets and retains replay for subsequent updates.

`policy_optimization_enabled: false`, `policy_optimization_weight: 0.0`,
`exploration_bonus_enabled: false`, `exploration_bonus_weight: 0.0`, and
`discrepancy_beta: 0.0` make the disabled auxiliaries explicit. Enabling the
M3PO-style auxiliaries is currently rejected because that optional path is not
implemented. The retained `on_policy` mode has its separate PPO/discrepancy
settings. See [`docs/IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md)
for the differences from the external implementation and legacy mode.

## Development checks

```bash
python -m compileall -q m4po tests scripts
python -m pytest -q
bash -n setup_venv.sh scripts/*.sh
RUN_DIR=/tmp/m4po_mock_smoke bash scripts/smoke_test_mock.sh
```

See [`docs/IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md) for the
equation-to-code map and [`docs/DEVELOPMENT.md`](DEVELOPMENT.md) for
invariants and integration contracts.

## License

MIT. See [`LICENSE`](../LICENSE).
