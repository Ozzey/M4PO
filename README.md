# M4PO

Reference implementation of **Massively Multi-Task Multi-Embodiment Model-Based
Policy Optimization (M4PO)**.

This repository follows the small, inspectable layout of the bundled
[`external/M3PO`](external/M3PO) implementation while implementing the M4PO
method described in [`docs/M4PO_CoRL_26-1.pdf`](docs/M4PO_CoRL_26-1.pdf). It is
not a rename of M3PO: M4PO uses fresh on-policy rollouts, two state-value
critics, a hierarchical task/embodiment latent model, a termination model, and
PPO updates on the likelihood of the *executed stochastic planner*.

## Included

- separate RGB-D, proprioception, structured-state, task-language, and
  embodiment encoders;
- hierarchical dynamics with embodiment prediction before task progression;
- frozen precomputed language features or a learned task-ID ablation;
- normalized, embodiment-masked continuous action tokenization with arbitrary
  shared-coordinate mappings;
- per-embodiment control-rate alignment and discounted low-level reward
  aggregation;
- EMA target encoder and world-model value head;
- terminal-masked finite-horizon world-model targets and losses;
- actor-sampled stochastic latent MPPI with a fitted first-action Gaussian;
- exact fixed-noise planner likelihood recomputation for PPO;
- distinct extrinsic and discrepancy-augmented model-free critics;
- fresh-rollout GAE, clipped PPO, discrepancy annealing, strict checkpoints,
  resume, JSONL metrics, and task × embodiment evaluation;
- a lazy sequential adapter for compatible prebuilt IsaacLab tasks, including
  ready-to-run three-task Forge training and smoke configurations;
- a dependency-free heterogeneous mock environment for complete CPU training,
  testing, checkpointing, and evaluation.

The bundled IsaacLab adapter trains across compatible public Gym registrations
sequentially. The included configuration cycles the prebuilt PegInsert,
GearMesh, and NutThread Forge tasks on their common Franka embodiment. The
paper's exact task assets, controllers, and hardware interfaces are not
publicly specified, so this is a runnable public-task integration rather than
a reproduction of its reported benchmark. Other simulators or task suites can
still be loaded with `isaaclab_factory: package.module:function` using the
contract in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

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

Resume only from a completed-boundary checkpoint (or from an interrupted
collection checkpoint, which is rolled back to its last completed update):

```bash
python -m m4po.train \
  --config m4po/configs/mock_smoke.yaml \
  --resume-checkpoint artifacts/runs/mock_smoke/checkpoints/latest.pt \
  --total-steps 128 \
  --log-dir artifacts/runs/mock_smoke_resumed
```

Standalone benchmarking defaults to the paper's 100 episodes per
task–embodiment pair. The shorter command above is intended only as a smoke
test; reported experiments should also aggregate independently trained seeds.

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

During training, all vector workers run one task for a complete fresh rollout.
At the next rollout boundary, a seeded shuffled deck selects a task; when the
selection changes, the adapter closes and reaps the current simulator child
before spawning one for the new task. Every three rollouts therefore cover all
three Forge tasks once in shuffled order. This is balanced sequential
multi-task collection, not simultaneous simulation of different registrations.

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

The supplied Forge optimizer settings are conservative engineering defaults,
not hyperparameters reported by the paper: one PPO epoch, actor learning rate
`3e-6`, and transactional target KL `0.02`. The longer configuration retains
256-transition planner minibatches to fit a 16 GB GPU; the smoke configuration
uses its complete 128-transition rollout. Post-step KL, clipping, raw
log-ratio saturation, accepted actor steps, retries, and early stopping are
recorded in `metrics.jsonl`.

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
rollout boundaries. The adapter aligns their timeout counting and adds the
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
configuration performs 64 rollouts, evenly split between the two tasks, and
the smoke path trains each task twice before evaluating the frozen checkpoint
on both registrations. The installed IsaacLab 2.3.0 registry and its 152 task
IDs are recorded in
[`docs/ISAACLAB_TASK_REGISTRY.md`](docs/ISAACLAB_TASK_REGISTRY.md).

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
| Agent, PPO, discrepancy, world-model objective, checkpoints | [`m4po/m4po.py`](m4po/m4po.py) |
| Multimodal hierarchical world model and EMA targets | [`m4po/common/world_model.py`](m4po/common/world_model.py) |
| Fixed-noise Gaussian actor | [`m4po/common/policy.py`](m4po/common/policy.py) |
| Stochastic differentiable MPPI | [`m4po/common/planner.py`](m4po/common/planner.py) |
| Fresh rollout storage | [`m4po/common/buffer.py`](m4po/common/buffer.py) |
| Action normalization/tokenization | [`m4po/common/action_tokenizer.py`](m4po/common/action_tokenizer.py) |
| GAE, lambda returns, masks, discrepancy | [`m4po/common/losses.py`](m4po/common/losses.py) |
| Online collection/update loop | [`m4po/trainer/online_trainer.py`](m4po/trainer/online_trainer.py) |
| Mock multi-task/multi-embodiment environment | [`m4po/envs/mock_env.py`](m4po/envs/mock_env.py) |
| Lazy IsaacLab factory loader | [`m4po/envs/isaaclab_env.py`](m4po/envs/isaaclab_env.py) |
| Sequential prebuilt IsaacLab adapter | [`m4po/envs/isaaclab_prebuilt.py`](m4po/envs/isaaclab_prebuilt.py) |
| Aggregate and worst-embodiment evaluation | [`m4po/common/evaluation.py`](m4po/common/evaluation.py) |
| Flat YAML/CLI configuration | [`m4po/common/config.py`](m4po/common/config.py) |

## Algorithm order

For iteration `k`, the trainer:

1. freezes the current world model `theta_k` and actor `psi_old` for collection;
2. stores each executed pre-tanh action, candidate noise, and old planner log
   likelihood in the fresh rollout `B_k`;
3. computes extrinsic and discrepancy-augmented terminal-masked GAE targets;
4. recomputes the planner Gaussian with `(theta_k, z_t, xi_t)` fixed and updates
   the actor and both critics with PPO;
5. updates the hierarchical world model only from `B_k`, then EMA-updates its
   target encoder and value head;
6. discards `B_k` and collects a new rollout.

No replay buffer, Q ensemble, elite-refit planner, or Q-maximizing policy loss
from M3PO is used in the M4PO learning path.

## Development checks

```bash
python -m compileall -q m4po tests scripts
python -m pytest -q
bash -n setup_venv.sh scripts/*.sh
RUN_DIR=/tmp/m4po_mock_smoke bash scripts/smoke_test_mock.sh
```

See [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) for the
equation-to-code map and [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for
invariants and integration contracts.

## License

MIT. See [`LICENSE`](LICENSE).
