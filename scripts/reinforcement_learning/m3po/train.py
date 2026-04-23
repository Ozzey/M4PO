"""Launch M3PO training against Isaac Lab tasks."""

import argparse
import importlib
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_SOURCE_DIRS = (
	REPO_ROOT / 'isaaclab' / 'source' / 'isaaclab',
	REPO_ROOT / 'isaaclab' / 'source' / 'isaaclab_assets',
	REPO_ROOT / 'isaaclab' / 'source' / 'isaaclab_tasks',
	REPO_ROOT / 'isaaclab' / 'source' / 'isaaclab_rl',
	REPO_ROOT / 'm4po_datacollection' / 'source' / 'm4po_datacollection',
)
M3PO_SOURCE_DIR = REPO_ROOT / 'isaaclab' / 'external' / 'm3po' / 'tdmpc2'


def configure_python_path() -> None:
	for path in reversed(LOCAL_SOURCE_DIRS + (M3PO_SOURCE_DIR,)):
		path_str = str(path)
		if path.is_dir() and path_str not in sys.path:
			sys.path.insert(0, path_str)


def has_override(overrides: list[str], key: str) -> bool:
	return any(arg == key or arg.startswith(f'{key}=') for arg in overrides)


parser = argparse.ArgumentParser(description='Train an M3PO agent with Isaac Lab.')
parser.add_argument(
	'--task',
	type=str,
	default='Template-M4po-DataCollection-G1-InspireFTP-Abs-v0',
	help='Isaac Lab gym task id to train on.',
)
parser.add_argument('--num_envs', type=int, default=None, help='Number of Isaac Lab environments.')
parser.add_argument('--seed', type=int, default=1, help='Random seed used by M3PO.')
parser.add_argument(
	'--obs',
	type=str,
	default='state',
	choices=('state',),
	help='Observation mode for the Isaac Lab M3PO backend.',
)
parser.add_argument(
	'--disable_fabric',
	action='store_true',
	default=False,
	help='Disable Fabric when constructing the Isaac Lab environment.',
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

configure_python_path()

# Register built-in Isaac Lab tasks and the local M4PO data-collection task package.
import isaaclab_tasks  # noqa: F401
import m4po_datacollection.tasks  # noqa: F401

# Provide probe-time context to the vendored M3PO Isaac Lab backend.
os.environ['M3PO_ISAACLAB_DEVICE'] = getattr(args_cli, 'device', None) or 'cuda:0'
os.environ['M3PO_ISAACLAB_USE_FABRIC'] = '0' if args_cli.disable_fabric else '1'

if not has_override(hydra_overrides, 'task'):
	hydra_overrides.append(f'task={args_cli.task}')
if args_cli.num_envs is not None and not has_override(hydra_overrides, 'num_envs'):
	hydra_overrides.append(f'num_envs={args_cli.num_envs}')
if args_cli.seed is not None and not has_override(hydra_overrides, 'seed'):
	hydra_overrides.append(f'seed={args_cli.seed}')
if not has_override(hydra_overrides, 'obs'):
	hydra_overrides.append(f'obs={args_cli.obs}')
if not has_override(hydra_overrides, 'compile'):
	hydra_overrides.append('compile=false')
if not has_override(hydra_overrides, 'enable_wandb'):
	hydra_overrides.append('enable_wandb=false')
if not has_override(hydra_overrides, 'exp_name'):
	hydra_overrides.append('exp_name=isaaclab_m3po')

sys.argv = [sys.argv[0]] + hydra_overrides

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
	m3po_train = importlib.import_module('train')
	m3po_train.launch()


if __name__ == '__main__':
	try:
		main()
	finally:
		simulation_app.close()
