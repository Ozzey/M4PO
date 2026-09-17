from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from m4po.common.config import M4POConfig, update_config_from_args
from m4po.trainer import OnlineTrainer


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    defaults = M4POConfig()
    type_hints = get_type_hints(M4POConfig)
    for field in fields(M4POConfig):
        name = field.name
        default = getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        kwargs = {"dest": name, "default": None, "help": f"Override config field `{name}`."}
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, **kwargs)
        elif default is None:
            field_type = type_hints.get(name, str)
            origin = get_origin(field_type)
            args = [item for item in get_args(field_type) if item is not type(None)]
            value_type = args[0] if origin is not None and args else str
            parser.add_argument(flag, type=value_type, **kwargs)
        else:
            parser.add_argument(flag, type=type(default), **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M4PO on fresh parallel rollouts.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="YAML config file.")
    add_config_args(parser)
    return parser.parse_args()


def train(cfg: M4POConfig) -> Path:
    return OnlineTrainer(cfg).train()


def main() -> None:
    args = parse_args()
    cfg = M4POConfig.from_yaml(args.config)
    cfg = update_config_from_args(cfg, args)
    checkpoint = train(cfg)
    print(json.dumps({"checkpoint": str(checkpoint), "log_dir": cfg.log_dir}, indent=2))


if __name__ == "__main__":
    main()

