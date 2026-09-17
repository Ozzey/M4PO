from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn


def set_seed(seed: int, torch_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(device)


def flatten_obs(obs: Mapping[str, Any]) -> np.ndarray:
    parts = []
    for key in sorted(obs):
        arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        parts.append(arr)
    if not parts:
        raise ValueError("Observation dict is empty")
    return np.concatenate(parts, axis=0).astype(np.float32)


def observation_to_torch(
    observation: Mapping[str, np.ndarray | torch.Tensor],
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in observation.items()
    }


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-12))


def linear_decay(start: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return start
    fraction = max(0.0, 1.0 - step / float(decay_steps))
    return start * fraction


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise ValueError("EMA source and target parameters do not match")
    for name, target_parameter in target_parameters.items():
        target_parameter.lerp_(source_parameters[name], 1.0 - decay)
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise ValueError("EMA source and target buffers do not match")
    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if target_buffer.is_floating_point():
            target_buffer.lerp_(source_buffer, 1.0 - decay)
        else:
            target_buffer.copy_(source_buffer)


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, data: Dict[str, Any]) -> None:
        payload = {"time": time.time(), **data}
        self._fh.write(
            json.dumps(payload, sort_keys=True, default=_json_default) + "\n"
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def latest_checkpoint(log_dir: str | Path) -> Optional[Path]:
    checkpoint_dir = Path(log_dir) / "checkpoints"
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()
