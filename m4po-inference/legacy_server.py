#!/usr/bin/env python3

"""Serve GR00T N1.5 checkpoints through the newer Isaac-GR00T policy protocol."""

from __future__ import annotations

import argparse
import importlib.machinery
import io
import os
from pathlib import Path
import sys
import types
from typing import Any

import msgpack
import numpy as np
import zmq


SCRIPT_DIR = Path(__file__).resolve().parent
N15_ROOT = SCRIPT_DIR / "vendor" / "Isaac-GR00T-n1.5-release-codeonly"
DEFAULT_DATA_CONFIG = "m4po_inference.legacy_data_config:G1ArenaGrootN15DataConfig"
HF_CACHE_ROOT = Path("/tmp/m4po-inference-hf-cache")


def _install_stub_modules() -> None:
    """Stub optional N1.5 dependencies that are not needed for inference here."""

    if "decord" not in sys.modules:
        decord = types.ModuleType("decord")
        decord.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
        decord.VideoReader = object
        decord.cpu = lambda *args, **kwargs: None
        sys.modules["decord"] = decord

    if "pytorch3d.transforms" not in sys.modules:
        pytorch3d = types.ModuleType("pytorch3d")
        pytorch3d.__spec__ = importlib.machinery.ModuleSpec("pytorch3d", loader=None)
        transforms = types.ModuleType("pytorch3d.transforms")
        transforms.__spec__ = importlib.machinery.ModuleSpec("pytorch3d.transforms", loader=None)

        def _unsupported(*args, **kwargs):
            raise RuntimeError("pytorch3d rotation conversion was requested but is unavailable.")

        for name in (
            "axis_angle_to_matrix",
            "matrix_to_axis_angle",
            "quaternion_to_matrix",
            "matrix_to_quaternion",
            "rotation_6d_to_matrix",
            "matrix_to_rotation_6d",
            "euler_angles_to_matrix",
            "matrix_to_euler_angles",
        ):
            setattr(transforms, name, _unsupported)

        pytorch3d.transforms = transforms
        sys.modules["pytorch3d"] = pytorch3d
        sys.modules["pytorch3d.transforms"] = transforms

    if "numpydantic" not in sys.modules:
        from pydantic_core import core_schema

        numpydantic = types.ModuleType("numpydantic")
        numpydantic.__spec__ = importlib.machinery.ModuleSpec("numpydantic", loader=None)

        class NDArray:
            @classmethod
            def __class_getitem__(cls, item):
                return cls

            @classmethod
            def __get_pydantic_core_schema__(cls, source_type, handler):
                return core_schema.any_schema()

            @classmethod
            def __get_pydantic_json_schema__(cls, schema, handler):
                return {"type": "array"}

        numpydantic.NDArray = NDArray
        sys.modules["numpydantic"] = numpydantic


def _configure_python_path() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    if str(N15_ROOT) not in sys.path:
        sys.path.insert(0, str(N15_ROOT))


def _configure_runtime_env() -> None:
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (HF_CACHE_ROOT / "hub").mkdir(parents=True, exist_ok=True)
    (HF_CACHE_ROOT / "modules").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
    os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_ROOT / "modules"))


def _patch_transformers_compat() -> None:
    import transformers
    import transformers.image_processing_utils_fast as image_processing_utils_fast
    import transformers.image_utils as image_utils
    from transformers.processing_utils import ProcessorMixin

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = image_utils.ImageInput

    if not hasattr(image_processing_utils_fast, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"):
        image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    if not hasattr(image_processing_utils_fast, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS"):
        image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""

    auto_processor_cls = transformers.AutoProcessor
    if getattr(auto_processor_cls, "_m4po_n15_patched", False):
        return

    original_from_pretrained = auto_processor_cls.from_pretrained

    def _patch_legacy_eagle_processor(processor_cls) -> None:
        if getattr(processor_cls, "_m4po_from_args_patched", False):
            return

        @classmethod
        def _compat_from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
            processor_dict = processor_dict.copy()
            return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

            processor_dict.pop("processor_class", None)
            processor_dict.pop("auto_map", None)
            processor_dict.update(kwargs)

            accepted_args = cls.__init__.__code__.co_varnames[: cls.__init__.__code__.co_argcount][
                1:
            ]
            validated = ProcessorMixin.validate_init_kwargs(
                processor_config=processor_dict,
                valid_kwargs=accepted_args,
            )
            if isinstance(validated, tuple):
                unused_kwargs, valid_kwargs = validated
            else:
                unused_kwargs = validated
                valid_kwargs = {
                    key: value
                    for key, value in processor_dict.items()
                    if key not in unused_kwargs
                }

            args_to_update = {
                index: valid_kwargs.pop(arg_name)
                for index, arg_name in enumerate(accepted_args)
                if arg_name in valid_kwargs and index < len(args)
            }
            args = [args_to_update.get(index, arg) for index, arg in enumerate(args)]

            processor = cls(*args, **valid_kwargs)
            if return_unused_kwargs:
                return processor, unused_kwargs
            return processor

        processor_cls.from_args_and_dict = _compat_from_args_and_dict
        processor_cls._m4po_from_args_patched = True

    @classmethod
    def _compat_auto_processor_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor_path = Path(pretrained_model_name_or_path).expanduser()
        if processor_path.is_dir() and (processor_path / "processing_eagle2_5_vl.py").is_file():
            from gr00t.model.backbone.eagle2_hg_model.processing_eagle2_5_vl import (
                Eagle2_5_VLProcessor,
            )

            _patch_legacy_eagle_processor(Eagle2_5_VLProcessor)
            kwargs.setdefault("trust_remote_code", True)
            kwargs.setdefault("use_fast", True)
            kwargs.setdefault("fix_mistral_regex", True)
            return Eagle2_5_VLProcessor.from_pretrained(
                str(processor_path),
                *args,
                **kwargs,
            )

        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    auto_processor_cls.from_pretrained = _compat_auto_processor_from_pretrained
    auto_processor_cls._m4po_n15_patched = True


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, obj, allow_pickle=False)
            return {"__ndarray__": True, "payload": buffer.getvalue()}
        return obj

    @staticmethod
    def _decode(obj: Any) -> Any:
        if isinstance(obj, dict) and obj.get("__ndarray__"):
            return np.load(io.BytesIO(obj["payload"]), allow_pickle=False)
        return obj


def _flatten_observation(observation: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    for key, value in observation.get("state", {}).items():
        flat[f"state.{key}"] = value
    for key, value in observation.get("video", {}).items():
        flat[f"video.{key}"] = value

    language = observation.get("language", {})
    if language:
        language_value = next(iter(language.values()))
        flat["annotation.human.action.task_description"] = language_value

    return flat


def _normalize_action_dict(action_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in action_dict.items():
        normalized[key[7:] if key.startswith("action.") else key] = value
    return normalized


def _serialize_modality_config(modality_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    serialized = {}
    for name, cfg in modality_config.items():
        serialized[name] = {
            "delta_indices": list(cfg.delta_indices),
            "modality_keys": list(cfg.modality_keys),
        }
    return serialized


class LegacyPolicyServer:
    def __init__(self, policy, host: str, port: int):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

    def _handle_get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
        flat_obs = _flatten_observation(observation)
        action_dict = self.policy.get_action(flat_obs)
        return [_normalize_action_dict(action_dict), {}]

    def _handle_get_modality_config(self):
        return _serialize_modality_config(self.policy.get_modality_config())

    def _handle_ping(self):
        return {"status": "ok", "message": "Server is running"}

    def _handle_reset(self, options: dict[str, Any] | None = None):
        return {}

    def _handle_kill(self):
        self.running = False
        return {"status": "ok"}

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")

        handlers = {
            "ping": lambda data: self._handle_ping(),
            "kill": lambda data: self._handle_kill(),
            "reset": lambda data: self._handle_reset(**data),
            "get_modality_config": lambda data: self._handle_get_modality_config(),
            "get_action": lambda data: self._handle_get_action(**data),
        }

        while self.running:
            try:
                request = MsgSerializer.from_bytes(self.socket.recv())
                endpoint = request.get("endpoint", "get_action")
                data = request.get("data", {})
                if endpoint not in handlers:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                response = handlers[endpoint](data)
                self.socket.send(MsgSerializer.to_bytes(response))
            except Exception as exc:  # noqa: BLE001
                print(f"Error in server: {exc}")
                self.socket.send(MsgSerializer.to_bytes({"error": str(exc)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GR00T N1.5 checkpoint via the modern policy API.")
    parser.add_argument("--model-path", required=True, help="Path to the local checkpoint directory.")
    parser.add_argument(
        "--embodiment-tag",
        default="new_embodiment",
        help="GR00T embodiment tag. The public Arena G1 checkpoint uses `new_embodiment`.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for inference.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the policy server to.")
    parser.add_argument("--port", type=int, default=5555, help="Port to bind the policy server to.")
    parser.add_argument("--denoising-steps", type=int, default=4, help="Number of action denoising steps.")
    parser.add_argument(
        "--data-config",
        default=DEFAULT_DATA_CONFIG,
        help="N1.5 data config to use. Defaults to the local Arena G1 compatibility config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not N15_ROOT.is_dir():
        raise FileNotFoundError(f"GR00T N1.5 sidecar checkout not found: {N15_ROOT}")

    _install_stub_modules()
    _configure_runtime_env()
    _configure_python_path()
    _patch_transformers_compat()

    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    data_config = load_data_config(args.data_config)

    print("Starting GR00T N1.5 compatibility server...")
    print(f"  Embodiment tag: {args.embodiment_tag}")
    print(f"  Model path: {model_path}")
    print(f"  Device: {args.device}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Data config: {args.data_config}")

    policy = Gr00tPolicy(
        model_path=str(model_path),
        embodiment_tag=args.embodiment_tag,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=args.denoising_steps,
        device=args.device,
    )

    server = LegacyPolicyServer(policy=policy, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
