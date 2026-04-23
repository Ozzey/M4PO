# M4PO Data Collection

This repository is an external Isaac Lab project focused on replaying HDF5 demonstrations with a local mirror of the official G1 Inspire pick-place environment.

Registered task:

- `Template-M4po-DataCollection-G1-InspireFTP-Abs-v0`

The local env entry point is:

- `source/m4po_datacollection/m4po_datacollection/tasks/manager_based/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py`

It wraps the official Isaac Lab config for:

- `Isaac-PickPlace-G1-InspireFTP-Abs-v0`

## Setup

Install the extension into the Isaac Lab conda environment:

```bash
source /home/aditya/miniconda3/etc/profile.d/conda.sh
conda activate m4po_isaaclab
python -m pip install -e source/m4po_datacollection
```

## Run

List the local environment:

```bash
./scripts/list_envs.py --keyword DataCollection
```

Replay the downloaded dataset:

```bash
./scripts/replay_dataset.py --device cpu --headless
```

Replay only a specific episode:

```bash
./scripts/replay_dataset.py --device cpu --headless --select_episodes 0
```

Optional sanity checks:

```bash
./scripts/random_agent.py --device cpu --headless
./scripts/zero_agent.py --device cpu --headless
```

## Notes

- The default dataset target is `/home/aditya/Desktop/Projects/M4PO/isaaclab/datasets/dataset_annotated_g1_locomanip.hdf5`.
- The downloaded dataset contains 32-D teleoperation actions, while the recreated G1 Inspire pick-place env uses a different action interface.
- Because of that mismatch, `scripts/replay_dataset.py` defaults to `--mode auto` and falls back to direct state-trajectory playback.
- If your session does not expose a CUDA device, append `--device cpu`.
