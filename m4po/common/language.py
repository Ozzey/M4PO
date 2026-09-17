from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def load_task_contexts(path: str | Path, task_names: Sequence[str]) -> np.ndarray:
    """Load frozen task-language features in the configured task order.

    ``.npy`` files contain a ``[num_tasks, context_dim]`` array. ``.npz``
    files may either contain that array under ``features`` or one vector per
    task name. Torch checkpoints may contain a tensor directly or a mapping
    with a ``features`` entry.
    """

    context_path = Path(path)
    if not context_path.exists():
        raise FileNotFoundError(f"Task-context file not found: {context_path}")
    if context_path.suffix == ".npy":
        value = np.load(context_path, allow_pickle=False)
    elif context_path.suffix == ".npz":
        archive = np.load(context_path, allow_pickle=False)
        if "features" in archive:
            value = archive["features"]
        else:
            missing = [name for name in task_names if name not in archive]
            if missing:
                raise ValueError(f"Task-context archive is missing tasks: {missing}")
            value = np.stack([archive[name] for name in task_names], axis=0)
    elif context_path.suffix in {".pt", ".pth"}:
        payload = torch.load(context_path, map_location="cpu", weights_only=False)
        value = payload.get("features") if isinstance(payload, dict) else payload
        if value is None:
            raise ValueError("Torch task-context checkpoint has no `features` entry")
        value = torch.as_tensor(value).cpu().numpy()
    else:
        raise ValueError("Task contexts must be stored as .npy, .npz, .pt, or .pth")
    features = np.asarray(value, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(task_names):
        raise ValueError(
            "Task contexts must have shape [num_tasks, context_dim], got "
            f"{features.shape} for {len(task_names)} tasks"
        )
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return (features / np.maximum(norms, 1e-8)).astype(np.float32)


def encode_task_texts_clip(
    task_texts: Sequence[str],
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
) -> np.ndarray:
    """Encode instructions with a frozen OpenCLIP text tower.

    OpenCLIP is deliberately an optional, lazy dependency. Production runs can
    precompute these features once and then use :func:`load_task_contexts`, so
    training workers never download weights or construct a text model.
    """

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "OpenCLIP is not installed. Install `m4po[clip]`, or provide a "
            "precomputed task-context file."
        ) from exc
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        tokens = tokenizer(list(task_texts)).to(device)
        features = model.encode_text(tokens).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return features.cpu().numpy().astype(np.float32)

