from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

IMPLEMENTATION_ID = "m4po_multitask_multiembodiment"
CHECKPOINT_SCHEMA_VERSION = 3


def _split_names(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class M4POConfig:
    """Flat configuration for replay learning, world modeling, and planning."""

    # Environment and multimodal observations.
    env: str = "mock"
    task: str = "reach"
    tasks: Optional[str] = None
    embodiment: str = "mock-arm"
    embodiments: Optional[str] = None
    multitask: bool = False
    multiembodiment: bool = False
    num_tasks: int = 1
    num_embodiments: int = 1
    use_images: bool = True
    image_size: int = 64
    image_channels: int = 4
    proprio_dim: int = 16
    state_dim: int = 16
    seed: int = 0
    num_envs: int = 8
    max_episode_steps: int = 100
    control_decimation: int = 1
    control_repeats: Optional[str] = None
    low_level_discount: Optional[float] = 1.0
    task_context_mode: str = "frozen"
    task_context_path: Optional[str] = None
    isaaclab_factory: Optional[str] = None
    isaaclab_device: str = "cuda:0"
    isaaclab_headless: bool = True
    isaaclab_enable_cameras: bool = False
    isaaclab_use_fabric: bool = True
    mmbench_root: Optional[str] = None
    mmbench_task_set: str = "soup"
    mmbench_eval_num_envs: int = 1
    mmbench_sampling: str = "episodes"

    # Episodic off-policy learning; PPO remains an explicit legacy mode.
    learning_mode: str = "off_policy"
    seed_steps: int = 1_000
    pretrain_updates: int = 1_000
    updates_per_step: float = 1.0
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    num_q: int = 5
    rho: float = 0.5
    total_steps: int = 1_048_576
    rollout_steps: int = 128
    updates_per_rollout: int = 4
    ppo_epochs: int = 4
    minibatch_size: int = 256
    eval_every: int = 131_072
    eval_at_start: bool = False
    eval_episodes: int = 3
    save_every: int = 131_072
    log_every: int = 16_384
    log_dir: str = "runs/m4po"

    # Multimodal encoder and hierarchical world model.
    model_horizon: int = 8
    planning_horizon: int = 5
    task_latent_dim: int = 128
    body_latent_dim: int = 128
    task_context_dim: int = 64
    embodiment_context_dim: int = 32
    encoder_dim: int = 256
    mlp_dim: int = 512
    num_encoder_layers: int = 2
    simnorm_dim: int = 8
    dropout: float = 0.01
    world_model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    grad_clip_norm: float = 20.0
    ema_decay: float = 0.99
    wm_lambda: float = 0.95
    consistency_coef: float = 20.0
    reward_coef: float = 1.0
    value_coef: float = 1.0
    termination_coef: float = 1.0

    # On-policy actor-critic optimization.
    discount: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.2
    ppo_target_kl: Optional[float] = None
    value_clip_ratio: Optional[float] = None
    critic_coef: float = 1.0
    entropy_coef: float = 1e-4

    # Stochastic MPPI distribution.
    planner_samples: int = 64
    inverse_temperature: float = 1.0
    min_std: float = 0.05
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # No fresh-transition auxiliary or intrinsic reward in off-policy mode.
    policy_optimization_enabled: bool = False
    policy_optimization_weight: float = 0.0
    exploration_bonus_enabled: bool = False
    exploration_bonus_weight: float = 0.0
    # Legacy on-policy model/model-free value-discrepancy bonus.
    discrepancy_beta: float = 0.0
    discrepancy_max: float = 2.0
    discrepancy_decay_fraction: float = 1.0
    normalization_epsilon: float = 1e-6

    # Runtime and checkpointing.
    device: str = "auto"
    torch_deterministic: bool = False
    matmul_precision: str = "highest"
    quiet: bool = False
    resume_checkpoint: Optional[str] = None
    init_checkpoint: Optional[str] = None
    save_replay: bool = False
    max_wall_time_seconds: int = 0

    @property
    def rollout_batch_size(self) -> int:
        return self.rollout_steps * self.num_envs

    @property
    def model_latent_dim(self) -> int:
        return self.task_latent_dim + self.body_latent_dim

    @property
    def latent_dim(self) -> int:
        """Compatibility name for the concatenated hierarchical latent."""

        return self.model_latent_dim

    @property
    def horizon(self) -> int:
        """Compatibility name for the world-model training horizon."""

        return self.model_horizon

    @property
    def num_samples(self) -> int:
        return self.planner_samples

    @property
    def enc_dim(self) -> int:
        return self.encoder_dim

    @property
    def num_enc_layers(self) -> int:
        return self.num_encoder_layers

    @property
    def z_coef(self) -> float:
        return self.consistency_coef

    @property
    def r_coef(self) -> float:
        return self.reward_coef

    @property
    def v_coef(self) -> float:
        return self.value_coef

    @property
    def d_coef(self) -> float:
        return self.termination_coef

    @property
    def task_names(self) -> List[str]:
        names = _split_names(self.tasks)
        return names if self.multitask and names else [self.task]

    @property
    def embodiment_names(self) -> List[str]:
        names = _split_names(self.embodiments)
        return names if self.multiembodiment and names else [self.embodiment]

    @property
    def episode_length(self) -> int:
        return self.max_episode_steps

    @property
    def control_repeat_values(self) -> List[int] | None:
        if self.control_repeats is None:
            return None
        entries = [item.strip() for item in self.control_repeats.split(",")]
        if not entries or any(not item for item in entries):
            raise ValueError("control_repeats must be comma-separated integers")
        return [int(item) for item in entries]

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self.image_channels, self.image_size, self.image_size)

    def validate(self) -> None:
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be highest, high, or medium")
        if self.max_wall_time_seconds < 0:
            raise ValueError("max_wall_time_seconds must be non-negative")
        if self.env.lower().strip() == "mmbench":
            from m4po.envs.mmbench_env import configure_mmbench

            configure_mmbench(self)
            if self.mmbench_eval_num_envs <= 0:
                raise ValueError("mmbench_eval_num_envs must be positive")
        if self.learning_mode not in {"off_policy", "on_policy"}:
            raise ValueError("learning_mode must be one of: off_policy, on_policy")
        if self.init_checkpoint and (
            self.learning_mode != "off_policy" or self.env != "mmbench"
        ):
            raise ValueError("Demonstration initialization requires off-policy MMBench")
        for name in ("seed_steps", "pretrain_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("replay_capacity", "batch_size", "num_q"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_q < 2:
            raise ValueError("num_q must be at least two")
        if not math.isfinite(self.updates_per_step) or self.updates_per_step <= 0:
            raise ValueError("updates_per_step must be finite and positive")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        if (
            self.policy_optimization_enabled
            or self.policy_optimization_weight != 0.0
            or self.exploration_bonus_enabled
            or self.exploration_bonus_weight != 0.0
        ):
            raise ValueError(
                "Fresh-transition policy auxiliaries and exploration bonuses are not "
                "implemented in the replay path; leave their flags false and weights zero"
            )
        if self.learning_mode == "off_policy" and self.discrepancy_beta != 0.0:
            raise ValueError(
                "Off-policy learning requires discrepancy_beta=0 (extrinsic rewards only)"
            )
        positive = {
            "eval_episodes": self.eval_episodes,
            "num_envs": self.num_envs,
            "max_episode_steps": self.max_episode_steps,
            "control_decimation": self.control_decimation,
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "updates_per_rollout": self.updates_per_rollout,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
            "model_horizon": self.model_horizon,
            "planning_horizon": self.planning_horizon,
            "task_latent_dim": self.task_latent_dim,
            "body_latent_dim": self.body_latent_dim,
            "task_context_dim": self.task_context_dim,
            "embodiment_context_dim": self.embodiment_context_dim,
            "encoder_dim": self.encoder_dim,
            "mlp_dim": self.mlp_dim,
            "num_encoder_layers": self.num_encoder_layers,
            "simnorm_dim": self.simnorm_dim,
            "planner_samples": self.planner_samples,
        }
        if self.use_images:
            positive.update(
                image_size=self.image_size, image_channels=self.image_channels
            )
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("proprio_dim", "state_dim"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.proprio_dim + self.state_dim <= 0 and not self.use_images:
            raise ValueError("At least one observation modality must be enabled")
        for name in ("eval_every", "save_every", "log_every"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not self.env.strip() or not self.task.strip() or not self.embodiment.strip():
            raise ValueError("env, task, and embodiment must be non-empty")

        task_names = _split_names(self.tasks)
        if self.multitask:
            if len(task_names) < 2:
                raise ValueError(
                    "multitask=True requires at least two comma-separated tasks"
                )
            if len(set(task_names)) != len(task_names):
                raise ValueError("Configured task names must be unique")
            if self.num_tasks == 1:
                self.num_tasks = len(task_names)
            elif self.num_tasks != len(task_names):
                raise ValueError("num_tasks must match the number of configured tasks")
        elif self.num_tasks != 1:
            raise ValueError("num_tasks must be 1 when multitask=False")

        embodiment_names = _split_names(self.embodiments)
        if self.multiembodiment:
            if len(embodiment_names) < 2:
                raise ValueError(
                    "multiembodiment=True requires at least two comma-separated "
                    "embodiments"
                )
            if len(set(embodiment_names)) != len(embodiment_names):
                raise ValueError("Configured embodiment names must be unique")
            if self.num_embodiments == 1:
                self.num_embodiments = len(embodiment_names)
            elif self.num_embodiments != len(embodiment_names):
                raise ValueError(
                    "num_embodiments must match the number of configured embodiments"
                )
        elif self.num_embodiments != 1:
            raise ValueError("num_embodiments must be 1 when multiembodiment=False")

        if self.control_repeats is not None:
            try:
                repeats = self.control_repeat_values
            except ValueError as exc:
                raise ValueError(
                    "control_repeats must be comma-separated integers"
                ) from exc
            assert repeats is not None
            if len(repeats) != self.num_embodiments:
                raise ValueError(
                    "control_repeats must contain one value per configured embodiment"
                )
            if any(value <= 0 for value in repeats):
                raise ValueError("control_repeats values must be positive")
        if self.task_context_path is not None and not self.task_context_path.strip():
            raise ValueError("task_context_path must be non-empty when configured")
        if self.task_context_mode not in {"frozen", "learned", "none"}:
            raise ValueError("task_context_mode must be one of: frozen, learned, none")
        if self.task_context_mode != "frozen" and self.task_context_path is not None:
            raise ValueError(
                "task_context_path is only valid when task_context_mode='frozen'"
            )
        if self.isaaclab_factory is not None and not self.isaaclab_factory.strip():
            raise ValueError("isaaclab_factory must be non-empty when configured")
        if not self.isaaclab_device.strip():
            raise ValueError("isaaclab_device must be non-empty")
        if (
            self.env.lower().strip() in {"isaaclab", "isaac_lab"}
            and not self.isaaclab_factory
        ):
            raise ValueError("env='isaaclab' requires an isaaclab_factory import path")

        if self.learning_mode == "off_policy" and self.total_steps % self.num_envs:
            raise ValueError("Off-policy total_steps must be divisible by num_envs")
        if (
            self.learning_mode == "on_policy"
            and self.total_steps % self.rollout_batch_size
        ):
            raise ValueError(
                "total_steps must be divisible by num_envs * rollout_steps so every "
                "update uses one complete fresh rollout"
            )
        if (
            self.learning_mode == "on_policy"
            and self.model_horizon > self.rollout_steps
        ):
            raise ValueError("model_horizon cannot exceed rollout_steps")
        if (
            self.learning_mode == "on_policy"
            and self.minibatch_size > self.rollout_batch_size
        ):
            raise ValueError(
                "minibatch_size cannot exceed the fresh rollout batch size"
            )
        if (
            self.learning_mode == "on_policy"
            and self.rollout_batch_size % self.minibatch_size
        ):
            raise ValueError(
                "num_envs * rollout_steps must be divisible by minibatch_size"
            )
        if self.task_latent_dim % self.simnorm_dim:
            raise ValueError("task_latent_dim must be divisible by simnorm_dim")
        if self.body_latent_dim % self.simnorm_dim:
            raise ValueError("body_latent_dim must be divisible by simnorm_dim")

        for name in ("world_model_lr", "actor_lr", "critic_lr", "grad_clip_norm"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 <= self.wm_lambda <= 1.0:
            raise ValueError("wm_lambda must be in [0, 1]")
        for name in (
            "consistency_coef",
            "reward_coef",
            "value_coef",
            "termination_coef",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not any(
            value > 0
            for value in (
                self.consistency_coef,
                self.reward_coef,
                self.value_coef,
                self.termination_coef,
            )
        ):
            raise ValueError(
                "At least one world-model loss coefficient must be positive"
            )

        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if (
            self.low_level_discount is not None
            and not 0.0 <= self.low_level_discount <= 1.0
        ):
            raise ValueError("low_level_discount must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < self.ppo_clip_ratio < 1.0:
            raise ValueError("ppo_clip_ratio must be in (0, 1)")
        if self.ppo_target_kl is not None and self.ppo_target_kl <= 0.0:
            raise ValueError("ppo_target_kl must be positive when configured")
        if self.value_clip_ratio is not None and not 0.0 < self.value_clip_ratio < 1.0:
            raise ValueError("value_clip_ratio must be in (0, 1) when configured")
        if self.critic_coef < 0 or self.entropy_coef < 0:
            raise ValueError("critic_coef and entropy_coef must be non-negative")

        if self.inverse_temperature <= 0:
            raise ValueError("inverse_temperature must be positive")
        if self.min_std <= 0:
            raise ValueError("min_std must be positive")
        if self.log_std_max <= self.log_std_min:
            raise ValueError("log_std_max must exceed log_std_min")
        if self.discrepancy_beta < 0 or self.discrepancy_max < 0:
            raise ValueError(
                "discrepancy_beta and discrepancy_max must be non-negative"
            )
        if not 0.0 < self.discrepancy_decay_fraction <= 1.0:
            raise ValueError("discrepancy_decay_fraction must be in (0, 1]")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_checkpoint_dict(cls, data: dict[str, Any]) -> "M4POConfig":
        """Checkpoints predating the mode field always used fresh-rollout PPO."""

        values = dict(data)
        values.setdefault("learning_mode", "on_policy")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "M4POConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        valid = {field.name for field in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        config = cls(**data)
        config.validate()
        return config

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def update_config_from_args(cfg: M4POConfig, args: Any) -> M4POConfig:
    cfg_dict = cfg.to_dict()
    valid = set(cfg_dict)
    for key, value in vars(args).items():
        if key in valid and value is not None:
            cfg_dict[key] = value
    config = M4POConfig(**cfg_dict)
    config.validate()
    return config


def parse_task_specs(cfg: M4POConfig) -> List[str]:
    return list(cfg.task_names)


def parse_embodiment_specs(cfg: M4POConfig) -> List[str]:
    return list(cfg.embodiment_names)
