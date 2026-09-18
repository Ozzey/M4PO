from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from m4po.common.action_tokenizer import ActionTokenizer
from m4po.common.buffer import ObservationSpec


MAX_STATE_DIM = 128
MAX_ACTION_DIM = 16
TASK_CONTEXT_DIM = 512
Observation = dict[str, np.ndarray]


def _newt_root(value: str | Path | None) -> Path:
    configured = value or os.environ.get("M4PO_NEWT_ROOT")
    return (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[2] / "external" / "newt"
    ).resolve()


@lru_cache(maxsize=8)
def _read_catalog(root: str, metadata_mtime: int, tasks_mtime: int):
    del metadata_mtime, tasks_mtime
    path = Path(root)
    metadata_bytes = (path / "tasks.json").read_bytes()
    task_bytes = (path / "tdmpc2" / "common" / "__init__.py").read_bytes()
    metadata = json.loads(metadata_bytes)
    task_sets = None
    for node in ast.parse(task_bytes).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_SET"
            for target in node.targets
        ):
            task_sets = ast.literal_eval(node.value)
            break
    if not isinstance(task_sets, dict):
        raise ValueError("NEWT source does not define the official TASK_SET mapping")
    soup = task_sets.get("soup", [])
    if len(soup) != 200 or len(set(soup)) != 200:
        raise ValueError("NEWT MMBench TASK_SET['soup'] must contain exactly 200 tasks")
    if any(task not in metadata for task in soup):
        raise ValueError("NEWT tasks.json is missing an official MMBench task")
    digest = hashlib.sha256(metadata_bytes + task_bytes).hexdigest()
    return metadata, task_sets, digest


def load_mmbench_catalog(root: str | Path | None = None):
    """Read official task metadata without importing any simulator packages."""

    path = _newt_root(root)
    try:
        return _read_catalog(
            str(path),
            (path / "tasks.json").stat().st_mtime_ns,
            (path / "tdmpc2" / "common" / "__init__.py").stat().st_mtime_ns,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"NEWT MMBench checkout is missing at {path}. Initialize external/newt "
            "or set mmbench_root / M4PO_NEWT_ROOT to the pinned NEWT checkout."
        ) from exc


def _native_source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    source = root / "tdmpc2" / "envs"
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(source).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _selection(cfg, metadata, task_sets) -> list[str]:
    configured = getattr(cfg, "tasks", None)
    if configured:
        names = (
            [name.strip() for name in configured.split(",") if name.strip()]
            if isinstance(configured, str)
            else list(configured)
        )
    else:
        task_set = str(getattr(cfg, "mmbench_task_set", "soup"))
        if task_set not in task_sets:
            raise ValueError(f"Unknown NEWT MMBench task set {task_set!r}")
        names = list(task_sets[task_set])
    if not names or len(set(names)) != len(names):
        raise ValueError("MMBench tasks must be non-empty and unique")
    official = set(task_sets["soup"])
    unknown = [name for name in names if name not in official or name not in metadata]
    if unknown:
        raise ValueError(
            f"Tasks are outside the official 200-task MMBench suite: {unknown}"
        )
    return names


def _embodiments(task_names, metadata):
    descriptions = list(
        dict.fromkeys(metadata[name]["embodiment"] for name in task_names)
    )
    names = [
        re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") for text in descriptions
    ]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Native embodiment descriptions do not have unique safe names")
    return names, descriptions


def configure_mmbench(cfg) -> None:
    """Resolve native task identities and dimensions before config validation."""

    if bool(cfg.use_images):
        raise ValueError(
            "The MMBench adapter currently supports native state observations only"
        )
    if int(cfg.control_decimation) != 1 or getattr(cfg, "control_repeats", None):
        raise ValueError(
            "MMBench action repeats are defined by the official NEWT wrappers"
        )
    if getattr(cfg, "low_level_discount", 1.0) not in (None, 1.0):
        raise ValueError(
            "MMBench preserves the official undiscounted native repeat rewards"
        )
    if getattr(cfg, "task_context_mode", "frozen") != "frozen" or getattr(
        cfg, "task_context_path", None
    ):
        raise ValueError("MMBench uses the official frozen tasks.json text embeddings")
    metadata, task_sets, _ = load_mmbench_catalog(getattr(cfg, "mmbench_root", None))
    task_names = _selection(cfg, metadata, task_sets)
    sampling = getattr(cfg, "mmbench_sampling", "episodes")
    if sampling not in {"episodes", "fixed"}:
        raise ValueError("mmbench_sampling must be 'episodes' or 'fixed'")
    if (
        sampling == "fixed"
        and not getattr(cfg, "_mmbench_evaluation", False)
        and int(cfg.num_envs) % len(task_names)
    ):
        raise ValueError(
            "Fixed MMBench sampling requires num_envs to be a multiple of num_tasks"
        )
    embodiment_names, _ = _embodiments(task_names, metadata)
    cfg.task = task_names[0]
    cfg.tasks = ",".join(task_names)
    cfg.multitask = len(task_names) > 1
    cfg.num_tasks = len(task_names)
    cfg.embodiment = embodiment_names[0]
    cfg.embodiments = ",".join(embodiment_names)
    cfg.multiembodiment = len(embodiment_names) > 1
    cfg.num_embodiments = len(embodiment_names)
    cfg.proprio_dim = 0
    cfg.state_dim = MAX_STATE_DIM
    cfg.task_context_dim = TASK_CONTEXT_DIM
    cfg.max_episode_steps = max(
        int(metadata[name]["max_episode_steps"]) for name in task_names
    )


def _load_native_module(root: Path, domain: str):
    """Load one upstream domain, bypassing NEWT's eager all-domain __init__."""

    native_path = root / "tdmpc2" / "envs"
    package = sys.modules.get("envs")
    if package is not None:
        existing = [Path(value).resolve() for value in getattr(package, "__path__", [])]
        if existing != [native_path]:
            raise RuntimeError(
                "The top-level 'envs' package is already owned by another checkout; "
                "run MMBench in a fresh process to avoid mixing native simulator code"
            )
    else:
        # Upstream wrappers use absolute `envs.*` imports. A namespace package
        # keeps those imports intact without executing their eager __init__.py.
        package = ModuleType("envs")
        package.__path__ = [str(native_path)]
        package.__package__ = "envs"
        package.__spec__ = importlib.machinery.ModuleSpec(
            "envs", loader=None, is_package=True
        )
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules["envs"] = package
    return importlib.import_module(f"envs.{domain}")


def _configure_native_asset_paths() -> None:
    """Honor ManiSkill's native setting before its import-time path binding.

    Older M4PO launchers used ``MANISKILL_ASSET_DIR``. The pinned ManiSkill
    release actually reads ``MS_ASSET_DIR`` and appends ``data`` itself. Keep
    the former as a compatibility alias without overriding an explicit native
    setting or modifying an existing user asset directory.
    """

    root = os.environ.get("MS_ASSET_DIR") or os.environ.get("MANISKILL_ASSET_DIR")
    if not root:
        return
    if not os.environ.get("MS_ASSET_DIR"):
        os.environ["MS_ASSET_DIR"] = root
    loaded = sys.modules.get("mani_skill")
    bound_path = getattr(loaded, "ASSET_DIR", None)
    if bound_path is not None and Path(bound_path).expanduser().resolve() != (
        Path(root).expanduser().resolve() / "data"
    ):
        raise RuntimeError(
            "ManiSkill was imported before MS_ASSET_DIR was configured. "
            "Set MS_ASSET_DIR to the .maniskill asset root (without /data) "
            "and start a fresh process; existing asset directories are unchanged."
        )


def make_native_mmbench_env(*, root: Path, task: str, domain: str, seed: int):
    """Instantiate the unmodified official per-domain state wrapper lazily."""

    module_name = "dmcontrol" if domain == "dmcontrol-ext" else domain
    try:
        _configure_native_asset_paths()
        module = _load_native_module(root, module_name)
        native_cfg = SimpleNamespace(
            task=task,
            seed=int(seed),
            obs="state",
            render_size=224,
            num_envs=1,
            child_env=True,
            save_video=False,
            rank=0,
        )
        return module.make_env(native_cfg)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot load official MMBench task {task!r} ({domain}). Install the "
            "pinned NEWT simulator dependencies; no substitute environment is used."
        ) from exc


class MMBenchVectorEnv:
    """State-only NEWT MMBench with bounded workers and episode-boundary rotation.

    All rewards, action repeats, time limits and native scores remain in the
    official wrappers. This layer only pads state/actions and schedules tasks.
    A shuffled deck visits each selected task before starting another sweep.
    """

    sequential_evaluation = True
    preserve_episodes_between_rollouts = True

    def __init__(self, cfg, *, native_factory: Callable[..., Any] | None = None):
        configure_mmbench(cfg)
        self.root = _newt_root(getattr(cfg, "mmbench_root", None))
        metadata, task_sets, digest = load_mmbench_catalog(self.root)
        self.task_names = list(cfg.task_names)
        self.embodiment_names, self.embodiment_descriptions = _embodiments(
            self.task_names, metadata
        )
        self.num_tasks = len(self.task_names)
        self.benchmark_task_count = 200
        self.benchmark_full_suite = set(self.task_names) == set(task_sets["soup"])
        self.num_embodiments = len(self.embodiment_names)
        self.num_envs = int(cfg.num_envs)
        self._sampling = getattr(cfg, "mmbench_sampling", "episodes")
        self._evaluation = bool(getattr(cfg, "_mmbench_evaluation", False))
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self._metadata = [metadata[name] for name in self.task_names]
        self.task_domains = [
            next(
                domain
                for domain, names in task_sets.items()
                if domain != "soup" and task in names
            )
            for task in self.task_names
        ]
        self.episode_lengths = np.asarray(
            [row["max_episode_steps"] for row in self._metadata], dtype=np.int64
        )
        self.max_episode_steps = int(self.episode_lengths.max())
        self.task_contexts = np.asarray(
            [row["text_embedding"] for row in self._metadata], dtype=np.float32
        )
        if (
            self.task_contexts.shape != (self.num_tasks, TASK_CONTEXT_DIM)
            or not np.isfinite(self.task_contexts).all()
        ):
            raise ValueError(
                "Official MMBench text embeddings must be finite 512-vectors"
            )
        self.task_embodiment_ids = np.asarray(
            [
                self.embodiment_descriptions.index(row["embodiment"])
                for row in self._metadata
            ],
            dtype=np.int64,
        )
        self.evaluation_pairs = [
            (task, int(body)) for task, body in enumerate(self.task_embodiment_ids)
        ]
        action_dims = []
        for description in self.embodiment_descriptions:
            dims = {
                int(row["action_dim"])
                for row in self._metadata
                if row["embodiment"] == description
            }
            if len(dims) != 1 or not 0 < next(iter(dims)) <= MAX_ACTION_DIM:
                raise ValueError(
                    "Native embodiment has inconsistent or oversized action dimensions"
                )
            action_dims.append(dims.pop())
        self.action_tokenizer = ActionTokenizer(
            [-np.ones(dim, dtype=np.float32) for dim in action_dims],
            [np.ones(dim, dtype=np.float32) for dim in action_dims],
            embodiment_names=self.embodiment_names,
            shared_action_dim=MAX_ACTION_DIM,
        )
        self.action_dim = MAX_ACTION_DIM
        self.action_low = -np.ones(self.action_dim, dtype=np.float32)
        self.action_high = np.ones(self.action_dim, dtype=np.float32)
        self.embodiment_action_dims = np.asarray(action_dims, dtype=np.int64)
        self.control_repeats = np.ones(self.num_embodiments, dtype=np.int64)
        self.low_level_discounts = np.ones(self.num_embodiments, dtype=np.float32)
        self.observation_spec = ObservationSpec(
            image_shape=(0, 0, 0), proprio_dim=0, state_dim=MAX_STATE_DIM
        )
        self.observation_shapes = self.observation_spec.shapes
        self.environment_signature = {
            "benchmark": "NEWT-MMBench-state",
            "catalog_sha256": digest,
            "native_sources_sha256": _native_source_digest(self.root),
            "embodiment_descriptions": self.embodiment_descriptions,
            "task_domains": self.task_domains,
            "task_embodiment_ids": self.task_embodiment_ids.tolist(),
            "native_episode_lengths": self.episode_lengths.tolist(),
            "native_action_repeats": "unchanged official domain wrappers",
            "score": "unchanged official info.score",
            "sampling": (
                "fixed task workers with equal transitions per task"
                if self._sampling == "fixed"
                else "shuffled task deck at episode boundaries"
            ),
        }
        self._factory = native_factory or make_native_mmbench_env
        self._base_seed = int(cfg.seed)
        self._rng = np.random.default_rng(self._base_seed)
        self._deck = np.empty(0, dtype=np.int64)
        self._cursor = 0
        self._pinned_task: int | None = 0 if self._evaluation else None
        self._generations = np.zeros(self.num_envs, dtype=np.int64)
        self._elapsed = np.zeros(self.num_envs, dtype=np.int64)
        self.task_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.embodiment_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.action_masks = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.envs: list[Any] = [None] * self.num_envs
        self._has_reset = False
        try:
            for worker in range(self.num_envs):
                task = (
                    worker % self.num_tasks
                    if self._sampling == "fixed" and not self._evaluation
                    else self._next_task()
                )
                self._replace_worker(worker, task)
        except BaseException:
            self.close()
            raise

    def _next_task(self) -> int:
        if self._pinned_task is not None:
            return self._pinned_task
        if self._cursor >= len(self._deck):
            self._deck = self._rng.permutation(self.num_tasks)
            self._cursor = 0
        task = int(self._deck[self._cursor])
        self._cursor += 1
        return task

    def _replace_worker(self, worker: int, task: int) -> None:
        previous = self.envs[worker]
        if previous is not None:
            previous.close()
            self.envs[worker] = None
        seed = int(
            np.random.SeedSequence(
                [self._base_seed, worker, task, int(self._generations[worker])]
            ).generate_state(1)[0]
        )
        native = self._factory(
            root=self.root,
            task=self.task_names[task],
            domain=self.task_domains[task],
            seed=seed,
        )
        self.envs[worker] = native
        expected_dim = int(self._metadata[task]["action_dim"])
        space = native.action_space
        if (
            tuple(space.shape) != (expected_dim,)
            or not np.allclose(space.low, -1)
            or not np.allclose(space.high, 1)
        ):
            raise ValueError(
                f"Official action space differs from metadata for {self.task_names[task]}"
            )
        self.task_ids[worker] = task
        self.embodiment_ids[worker] = self.task_embodiment_ids[task]
        self.action_masks[worker] = self.action_tokenizer.mask_for(
            int(self.embodiment_ids[worker])
        )
        self._elapsed[worker] = 0

    @staticmethod
    def _pad_observation(observation: Any) -> Observation:
        if isinstance(observation, Mapping):
            raise ValueError(
                "Expected native MMBench state vector, not an image/dictionary observation"
            )
        raw = np.asarray(observation, dtype=np.float32)
        if (
            raw.ndim != 1
            or not 0 < raw.size <= MAX_STATE_DIM
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                "MMBench native states must be finite vectors of at most 128 coordinates"
            )
        state = np.zeros(MAX_STATE_DIM, dtype=np.float32)
        state_mask = np.zeros(MAX_STATE_DIM, dtype=np.float32)
        state[: raw.size] = raw
        state_mask[: raw.size] = 1.0
        return {
            "image": np.empty((0, 0, 0), dtype=np.float32),
            "proprio": np.empty(0, dtype=np.float32),
            "proprio_mask": np.empty(0, dtype=np.float32),
            "state": state,
            "state_mask": state_mask,
        }

    @staticmethod
    def _stack(observations: Sequence[Observation]) -> Observation:
        return {
            key: np.stack([obs[key] for obs in observations]) for key in observations[0]
        }

    def reset(self) -> Observation:
        observations = []
        for worker, native in enumerate(self.envs):
            observation, _ = native.reset()
            observations.append(self._pad_observation(observation))
            self._elapsed[worker] = 0
        self._has_reset = True
        return self._stack(observations)

    reset_rollout = reset

    def set_evaluation_pair(self, task_id: int, embodiment_id: int) -> None:
        pair = (int(task_id), int(embodiment_id))
        if pair not in self.evaluation_pairs:
            raise ValueError(f"MMBench has no native task/embodiment pair {pair}")
        self._pinned_task = pair[0]
        self._has_reset = False
        for worker in range(self.num_envs):
            if int(self.task_ids[worker]) != pair[0]:
                self._generations[worker] += 1
                self._replace_worker(worker, pair[0])

    def step(self, actions: np.ndarray):
        if not self._has_reset:
            raise RuntimeError("Call reset() before step()")
        actions = np.asarray(actions, dtype=np.float32)
        if (
            actions.shape != (self.num_envs, self.action_dim)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("MMBench actions must be finite [num_envs, 16] arrays")
        actions = np.clip(actions, -1, 1) * self.action_masks
        observations, rewards, dones, infos = [], [], [], []
        for worker, native in enumerate(self.envs):
            task = int(self.task_ids[worker])
            embodiment = int(self.embodiment_ids[worker])
            local = self.action_tokenizer.detokenize(actions[worker], embodiment)
            raw, reward, terminated, truncated, native_info = native.step(local)
            self._elapsed[worker] += 1
            truncated = bool(truncated)
            terminated = bool(terminated)
            done = terminated or truncated
            if self._elapsed[worker] >= self.episode_lengths[task] and not done:
                raise RuntimeError(
                    f"Official MMBench task {self.task_names[task]!r} did not finish "
                    f"at its metadata horizon ({self.episode_lengths[task]} decisions); "
                    "refusing to replace the native time-limit semantics"
                )
            observation = self._pad_observation(raw)
            # In particular, preserve NaN/undefined success instead of bool(NaN).
            info = dict(native_info)
            info.update(
                {
                    "task_id": task,
                    "task_name": self.task_names[task],
                    "embodiment_id": embodiment,
                    "embodiment_name": self.embodiment_names[embodiment],
                    "terminated": terminated,
                    "truncated": truncated,
                    "timeout": truncated,
                    "control_repeat": 1,
                    "shared_action": actions[worker].copy(),
                    "local_normalized_action": local.copy(),
                }
            )
            if done:
                info["terminal_observation"] = {
                    key: value.copy() for key, value in observation.items()
                }
                self._generations[worker] += 1
                if self._sampling != "fixed" and self._pinned_task is None:
                    self._replace_worker(worker, self._next_task())
                raw, _ = self.envs[worker].reset()
                self._elapsed[worker] = 0
                observation = self._pad_observation(raw)
            observations.append(observation)
            rewards.append(float(reward))
            dones.append(done)
            infos.append(info)
        return (
            self._stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )

    def rollout_state_dict(self) -> dict[str, Any]:
        """Store the task sampler; simulator episodes restart when resuming."""

        return {
            "schema_version": 1,
            "rng": deepcopy(self._rng.bit_generator.state),
            "deck": self._deck.copy(),
            "cursor": self._cursor,
            "task_ids": self.task_ids.copy(),
            "generations": self._generations.copy(),
        }

    def load_rollout_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("Invalid MMBench sampler state")
        deck = np.asarray(state["deck"], dtype=np.int64)
        tasks = np.asarray(state["task_ids"], dtype=np.int64)
        generations = np.asarray(state["generations"], dtype=np.int64)
        cursor = int(state["cursor"])
        valid_deck = (
            deck.shape == (self.num_tasks,)
            and set(deck.tolist()) == set(range(self.num_tasks))
        ) or (self._sampling == "fixed" and deck.shape == (0,) and cursor == 0)
        if not valid_deck:
            raise ValueError("Invalid MMBench task deck")
        if (
            tasks.shape != (self.num_envs,)
            or np.any(tasks < 0)
            or np.any(tasks >= self.num_tasks)
        ):
            raise ValueError("Invalid MMBench worker task IDs")
        if self._sampling == "fixed" and not np.array_equal(
            tasks, np.arange(self.num_envs) % self.num_tasks
        ):
            raise ValueError(
                "Fixed MMBench sampler state changed its worker assignments"
            )
        if (
            generations.shape != (self.num_envs,)
            or np.any(generations < 0)
            or not 0 <= cursor <= self.num_tasks
        ):
            raise ValueError("Invalid MMBench sampler counters")
        self._rng.bit_generator.state = deepcopy(state["rng"])
        self._deck, self._cursor = deck.copy(), cursor
        self._generations = generations.copy()
        self._pinned_task = None
        self._has_reset = False
        for worker, task in enumerate(tasks):
            self._replace_worker(worker, int(task))

    state_dict = rollout_state_dict
    load_state_dict = load_rollout_state_dict

    def close(self) -> None:
        for worker, native in enumerate(self.envs):
            if native is not None:
                native.close()
                self.envs[worker] = None


def make_mmbench_vector_env(cfg) -> MMBenchVectorEnv:
    return MMBenchVectorEnv(cfg)


__all__ = [
    "MMBenchVectorEnv",
    "configure_mmbench",
    "load_mmbench_catalog",
    "make_mmbench_vector_env",
]
