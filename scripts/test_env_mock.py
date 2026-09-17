from __future__ import annotations

import argparse
import json

import numpy as np

from m4po.common.config import M4POConfig
from m4po.envs import make_vector_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the built-in M4PO mock environment.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = M4POConfig(
        multitask=True,
        tasks="reach,push",
        num_tasks=2,
        multiembodiment=True,
        embodiments="arm,hand",
        num_embodiments=2,
        num_envs=4,
        use_images=args.images,
        image_size=16,
        proprio_dim=6,
        state_dim=7,
        seed=args.seed,
    )
    env = make_vector_env(cfg)
    observation = env.reset()
    rng = np.random.default_rng(args.seed + 1)
    completed = 0
    try:
        for _ in range(args.steps):
            action = rng.uniform(-1.0, 1.0, (env.num_envs, env.action_dim)).astype(np.float32)
            observation, _, done, _ = env.step(action * env.action_masks)
            completed += int(done.sum())
    finally:
        env.close()
    print(
        json.dumps(
            {
                "observation_shapes": {name: list(value.shape) for name, value in observation.items()},
                "action_dim": env.action_dim,
                "action_masks": env.action_masks.tolist(),
                "task_ids": env.task_ids.tolist(),
                "embodiment_ids": env.embodiment_ids.tolist(),
                "control_repeats": env.control_repeats.tolist(),
                "completed_episodes": completed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

