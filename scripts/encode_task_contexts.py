from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from m4po.common.language import encode_task_texts_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen CLIP task contexts.")
    parser.add_argument("--texts", nargs="+", required=True, help="Task instructions in config order.")
    parser.add_argument("--out", required=True, help="Output .npy path.")
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = encode_task_texts_clip(
        args.texts,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )
    path = Path(args.out)
    if path.suffix != ".npy":
        raise ValueError("--out must end in .npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features, allow_pickle=False)
    print(f"saved {features.shape[0]} x {features.shape[1]} task contexts to {path}")


if __name__ == "__main__":
    main()

