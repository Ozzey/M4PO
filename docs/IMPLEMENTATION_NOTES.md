# M4PO implementation notes

## What changed from M3PO

The bundled M3PO project is used for structure and style only. Its off-policy
episodic replay, distributional Q ensemble, elite-refitted iterative MPPI,
planner warm start, and Q-maximizing actor loss are intentionally absent.
M4PO follows Algorithms 1–2 of the supplied paper:

- every update consumes one newly collected rollout batch;
- PPO optimizes the Gaussian fitted by stochastic MPPI, not the actor prior's
  direct likelihood;
- an extrinsic critic and an augmented critic are separate from the world-model
  value head;
- actor/critic updates happen before world-model updates, preserving the
  collection snapshot during planner likelihood recomputation.

## Multimodal hierarchy

`HierarchicalWorldModel` encodes RGB-D, masked proprioception, and masked
structured state separately. Task and embodiment contexts join those features
in a shared fusion network. Two SimNorm components form

```text
z_t = [g_t, b_t]
```

where `g_t` is task-level and `b_t` embodiment-level. Prediction is ordered:

```text
b_hat[t+1] = f_b(z_t, action_t, action_mask, embodiment_context)
g_hat[t+1] = f_g(z_t, b_hat[t+1], action_t, task_context)
```

Reward and termination heads receive latent, masked action, and both contexts;
the world-model value head receives latent and both contexts. The target
encoder and target value head are frozen copies updated only by EMA.

## World-model targets and masks

Each gradient step samples length-`model_horizon` segments from the current
fresh rollout. EMA value targets are recomputed and detached on every step:

```text
y_H = V_bar(z_bar_H)
y_s = r_s + gamma (1 - d_s)
      [(1 - lambda_wm) V_bar(z_bar_[s+1]) + lambda_wm y_[s+1]]
```

Continuation masks follow `M_0=1`, `M_[j+1]=M_j(1-d_j)`. Reward, value, and
termination losses use `M_j`, so the terminal transition is included. Latent
consistency uses `M_[j+1]`, excluding terminal/auto-reset boundaries. Every
segment is divided by its own valid count before the batch mean.

## Planner distribution and likelihood

Candidate noise has shape `[B, N, H_p, D_a]`. The actor reparameterizes all
candidates in pre-tanh space, the fixed world model scores their latent
rollouts with predicted continuation, and a stable softmax forms weights. The
weighted first-step candidates define `m_v` and diagonal variance with
`min_std**2` added.

Collection stores candidate noise, the sampled executed pre-tanh action, and
its old Gaussian log density. PPO recomputes every candidate under the current
actor while holding the collection world model, initial latent, and noise
fixed. Gaussian densities are summed only over valid action coordinates. The
shared tanh Jacobian and candidate-noise density therefore cancel in the ratio.

## Discrepancy and actor-critic targets

The one-step estimates are

```text
Q_MB = reward_hat + gamma (1 - done_hat) V_world(z_hat_next)
Q_MF = reward       + gamma (1 - done)     V_ext(z_next)
delta = abs(Q_MB - Q_MF)
```

`delta` is batch-normalized and clipped to `[0, discrepancy_max]`. Its linearly
annealed coefficient augments rewards only for the augmented critic and actor
advantage. Raw rewards remain unchanged for the extrinsic critic and every
world-model target.

Both critics use terminal-masked GAE with rollout-end bootstrapping. Returns
and advantages are detached once and held fixed across PPO epochs. The actor
uses normalized augmented advantages.

PPO likelihood diagnostics are recomputed after each accepted actor optimizer
step. `approximate_kl` and `clip_fraction` therefore describe the updated
planner policy, while `max_abs_log_ratio` and
`log_ratio_saturation_fraction` retain the unclamped log-ratio signal. When
`ppo_target_kl` is configured, an unsafe actor proposal is transactional: actor
parameters and Adam state are restored, the same gradient is retried with a
halved learning rate, and only an in-threshold proposal is accepted. If no
bounded retry is safe, later actor steps stop but all configured critic updates
still run. `max_observed_approximate_kl` includes accepted, rejected, and
pre-step checks so an accepted-only mean cannot hide the stopping event.

## Action and observation boundaries

`ActionTokenizer` normalizes physical coordinates, scatters arbitrary local
coordinates into a shared action vector, and zeroes invalid tokens. Environment
wrappers alone decode and denormalize physical commands. Masks are applied by
the actor, planner candidates, planner likelihood, dynamics, heads, rollout
storage, and execution.

Proprioception and structured state follow the same padding rule. Encoders see
both `value * mask` and the mask itself, making outputs invariant to arbitrary
data in invalid padded coordinates while distinguishing padding from a real
normalized zero.

## Checkpoints and resume

Checkpoints strictly identify the M4PO implementation/schema and contain the
resolved config, observation/action shapes, frozen language features, online
and EMA world-model state, actor, both critics, all three optimizers, global
random-number-generator state, and optional environment rollout-sampler
state. A resume always starts by collecting a new complete rollout; planner
noise and stale on-policy batches are never restored. Checkpoints captured
during a partially applied optimizer update are marked diagnostic-only and
cannot be resumed.
