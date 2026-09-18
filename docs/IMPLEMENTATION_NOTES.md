# M4PO implementation notes

## Default learning semantics

`learning_mode: off_policy` follows the main learning semantics of the bundled
M3PO checkout (`7a4cc83`): completed episodes enter replay, model and Q updates
sample fixed-horizon replay sequences, and the actor maximizes Q on detached
latents with a small entropy term. The same transition may be sampled by many
updates. Collection being online does not make those updates on-policy.

The explicit exploration bonus and PPO-like auxiliary are disabled:

```yaml
policy_optimization_enabled: false
policy_optimization_weight: 0.0
exploration_bonus_enabled: false
exploration_bonus_weight: 0.0
discrepancy_beta: 0.0
entropy_coef: 0.0001
```

The optional M3PO auxiliary paths are not implemented; enabling their flags or
weights is rejected. The retained legacy `learning_mode: on_policy` path has
its own PPO and discrepancy settings.

Uniform random actions seed replay. Once warmup has completed and replay is
sampleable, the trainer performs `pretrain_updates` once, then allocates
`updates_per_step` gradient updates per individual environment transition.
Fractional update budgets accumulate across vector steps. With 32
environments, `0.03125` schedules one update per vector step. The general
default is `1.0`, matching M3PO's update-per-transition convention; supplied
IsaacLab configs use `0.03125` as an explicit compute tradeoff.

## Differences from external M3PO

This is a hybrid implementation, not an exact reproduction. M4PO retains its
multimodal hierarchical model, embodiment-aware action masks, task contexts,
termination head, and stochastic MPPI planner. Reward and Q losses are scalar
regression losses rather than M3PO's distributional two-hot losses. The
planner fits a first-action Gaussian from actor-generated trajectories; it
does not adopt M3PO's iterative elite-refitting or planner warm start.

After warmup, stochastic candidates and sampled planner actions continue to
provide behavioral exploration even though no intrinsic exploration reward is
added to replay rewards, model targets, Q targets, or planner scores.

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
the legacy world-model value head receives latent and both contexts. The
off-policy path learns a Q ensemble conditioned on latent, action, and the
contexts, using frozen EMA target encoder and Q copies. Its legacy value head
is frozen and unused; the two legacy state-value critics are not constructed.

## Episodic replay and off-policy optimization

Replay stores completed episodes and samples contiguous length-`model_horizon`
sequences without crossing an episode, task, or embodiment boundary. True final
observations are preserved independently of the environment's auto-reset
observations. Terminal transitions participate in reward, Q, and termination
losses, with consistency targets taken from the true final observation. A true
termination stops Q bootstrapping; episode boundaries stop replay sequences
from reaching an auto-reset state. `rho` weights successive model rollout
losses.

Replay samples uniformly over retained transition starts. Short episodes and
end-of-episode tails are padded to the fixed horizon; a validity mask excludes
that padding from every model and actor loss. Unlike the reference M3PO buffer,
this allows short failure episodes to contribute before the policy can survive
a full model horizon.

Replay collection runs in `rollout_steps` blocks so the sequential IsaacLab
adapter can switch tasks. Unfinished episodes at a collection boundary are
discarded rather than committed as completed episodes. Use a block at least as
long as `max_episode_steps` when guaranteed complete-episode coverage matters,
and inspect `discarded_partial_transitions`. Task switching changes future
collection only; completed episodes from prior tasks remain sampleable.
MMBench sets `preserve_episodes_between_rollouts=True`, so its native episodes
continue across these collection blocks without a forced reset.

Each update fits dynamics consistency with mean squared error against the EMA
target encoder, and reward and Q predictions with scalar smooth-L1 losses.
Q targets use the online encoding of the true successor observation, a sampled
Gaussian actor action, and the minimum of two randomly selected target Q heads:

```text
y_t = r_t + gamma (1 - true_terminated_t) min(Q_bar_i, Q_bar_j)(z_[t+1], a_[t+1])
```

Time-limit truncations can bootstrap from their true final observations;
physical terminations cannot. As in the reference M3PO path, the Q target has
no entropy reward term. Each model rollout step is weighted by `rho**t` and its
validity mask.

Actor updates use detached model latents and frozen model, context, and Q
parameters, preserving the derivative of Q with respect to the actor's action.
They maximize the mean Q ensemble prediction, divided by a running scale, plus
the squashed Gaussian entropy term. The scale tracks the 5th–95th percentile
Q spread by EMA with a floor of one. The squashed Gaussian entropy sums valid
action dimensions and, following M3PO's `scaled_entropy`, is multiplied by the
number of valid dimensions before applying `entropy_coef: 0.0001`.
This is not a PPO ratio or a planner-likelihood
objective. EMA targets are updated after optimization. Raw extrinsic rewards
are used throughout this path.

## Legacy on-policy world-model targets

The remaining PPO/GAE sections describe `learning_mode: on_policy`, retained
for existing checkpoints and explicit comparisons. They do not control the
default off-policy optimizer.

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

## Planner distribution and legacy likelihood

Candidate noise has shape `[B, N, H_p, D_a]`. The actor reparameterizes all
candidates in pre-tanh space, the fixed world model scores their latent
rollouts with predicted continuation, and a stable softmax forms weights. The
weighted first-step candidates define `m_v` and diagonal variance with
`min_std**2` added.

In off-policy mode, the planner's terminal bootstrap is the online Q ensemble
mean for a sampled actor action. Legacy mode instead uses the state-value head.

Both learning modes use this fitted Gaussian for environment actions.
Off-policy updates do not use collection likelihoods. Legacy on-policy
collection stores candidate noise, the sampled executed pre-tanh action, and
its old Gaussian log density. PPO recomputes every candidate under the current
actor while holding the collection world model, initial latent, and noise
fixed. Gaussian densities are summed only over valid action coordinates. The
shared tanh Jacobian and candidate-noise density therefore cancel in the ratio.

## Legacy discrepancy and actor-critic targets

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
the actor, planner candidates, planner likelihood, dynamics, heads, replay and
rollout storage, and execution.

Proprioception and structured state follow the same padding rule. Encoders see
both `value * mask` and the mask itself, making outputs invariant to arbitrary
data in invalid padded coordinates while distinguishing padding from a real
normalized zero.

## Checkpoints and resume

Schema-3 checkpoints identify the learning mode and contain the resolved
config, observation/action shapes, frozen language features, online and EMA
model state, actor, mode-specific critics, optimizers, running Q scale,
training counters, random-number-generator state, and optional environment
sampler state.

With `save_replay: true`, `latest.pt` additionally saves completed replay
episodes and their sampling RNG. Historical `step_N.pt` checkpoints remain
lightweight. Without a replay snapshot, off-policy resume starts with empty
replay and random warmup; completed pretraining is not repeated. Even with a
snapshot, unfinished simulator episodes restart, so trajectories need not
match an uninterrupted run. `max_wall_time_seconds` stops safely between
optimizer updates, retaining outstanding update credit for continuation.
Atomic `latest_status.json` records whether both collection and all pending
updates have completed; reaching `total_steps` alone is not sufficient.
Legacy on-policy
resume collects a new complete rollout and never restores stale planner noise
or advantages. Checkpoints captured during a partially applied optimizer
update remain diagnostic-only.

Schema-2 checkpoints are interpreted as `learning_mode: on_policy`; their
previous discrepancy settings are preserved. They can be evaluated or resumed
in legacy mode, but cannot initialize an off-policy resume. Switching learning
modes requires a fresh training run.
