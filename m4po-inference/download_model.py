from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "assets" / "model" / "GN1x-Tuned-Arena-G1-Loco-Manipulation"
DEFAULT_ALLOW_PATTERNS = [
    "README.md",
    "config.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "experiment_cfg/**",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public Arena G1 GR00T checkpoint into m4po-inference.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model id to download.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the checkpoint should be staged.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {args.model_id} into {output_dir}")
    local_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
    )
    print(f"[INFO] Model ready at {local_path}")


if __name__ == "__main__":
    main()
