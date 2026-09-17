# Developer guide

## Invariants

1. Every optimization iteration uses one fresh `RolloutBatch`; never carry a
   transition, planner noise, or advantage into the next iteration.
2. Finish all planner-policy and critic updates before modifying the collection
   world model.
3. Recompute planner likelihoods with initial latents and candidate noise
   detached, world-model parameters frozen, and gradients enabled through the
   frozen dynamics into the actor.
4. Store and score pre-tanh executed actions. Do not add the tanh Jacobian to
   the PPO ratio because it is identical in numerator and denominator.
5. Apply action masks to actor samples, dynamics, prediction, likelihood, and
   execution. Invalid shared coordinates must be exactly zero.
6. Use raw extrinsic rewards for the extrinsic critic and world model. The
   discrepancy bonus may affect only augmented GAE/returns.
7. Include terminal transitions in reward/value/done losses; exclude them from
   latent consistency and prevent every target from crossing a reset.
8. Update only the target encoder and world-value target through EMA. Target
   parameters remain frozen.
9. A resumed run discards any interrupted rollout and collects a complete new
   batch before updating.
10. Keep simulator and CLIP imports lazy so mock tests run without either
    dependency.

## Environment contract

Vector environments expose:

```text
num_envs, num_tasks, num_embodiments, max_episode_steps
task_ids[num_envs], embodiment_ids[num_envs]
task_names, embodiment_names
action_dim, action_masks[num_envs, action_dim]
observation_spec, task_contexts[num_tasks, feature_dim]
reset() -> observation dict
reset_rollout() -> observation dict  # resample contexts, then reset
rollout_state_dict() / load_rollout_state_dict(state)  # optional resume state
step(shared_normalized_actions) -> (observation, reward, done, infos)
```

`task_ids`, `embodiment_ids`, and `action_masks` always describe the most
recently returned observation. Adapters may change them during `reset`,
`reset_rollout`, or when a completed worker is auto-reset, but never midway
through an unfinished episode. When an auto-reset also changes context, the
completed transition's `info` must retain its original `task_id` and
`embodiment_id`. Callers re-read metadata after every reset and step.

The observation dictionary has time/batch-leading float32 arrays and the five
keys `image`, `proprio`, `proprio_mask`, `state`, and `state_mask`. A blind
configuration represents image as shape `[..., 0, 0, 0]`. `step` auto-resets,
and a completed worker must put its true padded final observation under
`info["terminal_observation"]` plus a boolean `success`.

Agent actions are shared normalized tokens in `[-1,1]`. The wrapper owns local
coordinate extraction, denormalization, controller interpolation/repetition,
and reward/termination aggregation. This boundary permits an external
IsaacLab adapter without importing simulator types into the learning code.
If an adapter omits `reset_rollout`, the trainer falls back to `reset`; a
multi-task adapter should implement it so every fresh batch resamples balanced
task/embodiment assignments as required by Algorithm 2.

For deterministic boundary resume, adapters may implement
`rollout_state_dict` and `load_rollout_state_dict`; the trainer also recognizes
the conventional `state_dict` and `load_state_dict` aliases. The state must
restore context-sampling RNG/decks and the assignments that the next
`reset_rollout` will consume. Simulator state inside a discarded partial
rollout does not need to be checkpointed.

Evaluation requests `num_tasks * num_embodiments` workers and requires the
post-reset assignments to contain every Cartesian pair exactly once. Checkpoint
evaluation also compares the adapter, ordered names, observation and action
interfaces, control rates, episode limit, and task-context hash against the
saved environment signature before executing actions.

An adapter that cannot host all pairs concurrently may instead set
`sequential_evaluation = True` and implement
`set_evaluation_pair(task_id, embodiment_id)`. The evaluator then visits the
Cartesian pairs in order. After each selection, `reset()` and every subsequent
auto-reset must expose only that pair, and switching pairs must preserve
`num_envs`. Returns, lengths, and successes are accumulated under the same
per-pair reporting schema as concurrent evaluation.

## Rollout shapes

```text
observations:      dict values [T, B, ...]
next_observations: dict values [T, B, ...]
actions/masks:                 [T, B, D_a]
pre_tanh_actions:              [T, B, D_a]
candidate_noise:               [T, B, N, H_p, D_a]
old_log_prob:                  [T, B, 1]
rewards/terminated:            [T, B, 1]
task/embodiment IDs:           [T, B]
```

`next_observations` stores the true terminal observation even though the next
collection step starts from the auto-reset observation.

## Adding a simulator

Implement the environment contract in a simulator-specific factory callable,
then configure its import path as `isaaclab_factory`. The lazy loader in
`m4po/envs/isaaclab_env.py` validates the returned contract. Keep
simulator-specific observation selection, normalization statistics, control
coordinate maps, physical bounds, and control repeat values in that adapter.
Precompute CLIP vectors or expose them as `task_contexts`; never instantiate a
trainable text tower inside a simulation worker.

## Built-in sequential IsaacLab adapter

`m4po.envs.isaaclab_prebuilt:make_isaaclab_prebuilt_env` integrates compatible
public IsaacLab Gym registrations without importing Isaac Sim on mock-only
paths. The supplied Forge configs use:

```text
Isaac-Forge-PegInsert-Direct-v0
Isaac-Forge-GearMesh-Direct-v0
Isaac-Forge-NutThread-Direct-v0
```

These tasks share the Franka embodiment, a seven-dimensional normalized action
interface, simulation time step, control decimation of eight, and observation
layout. The adapter pads only IsaacLab's deployable `policy` group into
`proprio`; privileged `critic` observations are intentionally ignored and
`state_dim` is zero. It exposes no image observation. Actions are clipped to
`[-1,1]`, masked, and prefix-cropped to the selected task's local continuous
action dimension. It captures true terminal observations around IsaacLab's
auto-reset and reports Forge's `ep_succeeded` as episode success.

IsaacLab owns one global `SimulationContext` per process, and stock Forge tasks
cannot be safely reconstructed behind a previously used context. The adapter
therefore keeps one registration active for an entire fresh rollout inside an
isolated simulator child. A seeded shuffled deck selects every configured task
once per cycle; a task change closes and reaps the old child before spawning a
new child with its own SimulationApp and DirectRLEnv. Checkpoints preserve the
deck, sampler RNG, and reset counters, but intentionally omit simulator state
from a discarded partial rollout.

Set `eval_every: 0` for training with this adapter. In-process evaluation would
start another Isaac Sim worker while the training worker still owns GPU memory.
Run `m4po.evaluate` only after training exits and its simulator child has been
reaped. For evaluation, `sequential_evaluation = True` and
`set_evaluation_pair(task_id, 0)` cause all workers to evaluate one requested
Forge task before the adapter replaces that simulator child for the next task.

The adapter currently requires one common embodiment, continuous vector action
spaces, policy-vector observations without cameras, and identical `dt` and
decimation across all configured tasks. It excludes privileged critic
observations, does not make distinct Gym registrations coexist in one scene,
and child-process reconstruction makes task boundaries relatively expensive.

On the development host, run it with the IsaacLab interpreter:

```bash
export ISAACLAB_PYTHON=/home/aditya/miniconda3/envs/env_isaaclab/bin/python
bash scripts/smoke_test_isaaclab_forge.sh

"${ISAACLAB_PYTHON}" -m m4po.train \
  --config m4po/configs/isaaclab_forge.yaml
```

The smoke script performs training first and standalone evaluation second. It
may be redirected without editing the script:

```bash
RUN_DIR=/tmp/m4po_forge_smoke TOTAL_STEPS=768 EPISODES=1 \
  bash scripts/smoke_test_isaaclab_forge.sh
```

## Checks

```bash
python -m compileall -q m4po tests scripts
python -m pytest -q
bash -n setup_venv.sh scripts/*.sh
RUN_DIR=/tmp/m4po_mock_smoke bash scripts/smoke_test_mock.sh
RUN_DIR=/tmp/m4po_forge_smoke bash scripts/smoke_test_isaaclab_forge.sh
```
