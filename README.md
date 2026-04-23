# M4PO

<p align="center">
  <img src="docs/assets/g1_pick_place.jpg" alt="Unitree G1 dexterous pick-and-place in Isaac Lab" width="760">
</p>

M4PO is an Isaac Lab workspace for the G1 Dexterous Inspire Hand pick-and-place task. The project uses a GR00T-style environment interface for data collection, pretraining, and inference, and a modified NEWT world-model RL stack as the main research contribution.

## Naming Note

This repository still contains some upstream names:

- `GR00T` and `GROOT` appear in environment names, scripts, and vendored code. In this codebase, they should be read as placeholders for the M4PO-compatible multimodal interface.
- `NEWT` and the local `m3po` folder name appear in the RL backend because that code started from upstream NEWT. In this codebase, that backend is the M4PO training stack.

The project name and contribution are M4PO. The task focus is G1 + Inspire 5-finger pick-and-place.

## Repository Layout

- `isaaclab/`: Isaac Lab integration point, local task registration, GR00T-style task variants, and the M4PO launcher.
- `m4po_datacollection/`: local task package, dataset replay tools, and asset-backed data collection utilities.
- `m4po-inference/`: GR00T-compatible inference helpers, checkpoint download script, and policy-server launcher.
- `unitree_sim_isaaclab_ref/`: upstream Unitree reference repository kept around for comparison and asset provenance.

## Pipeline

1. Data collection uses the local G1 Inspire pick-and-place task package in `m4po_datacollection/`.
2. Pretraining and inference use GR00T-shaped observations and action adapters so public GR00T checkpoints and servers can talk to the task.
3. M4PO training uses the modified NEWT backend vendored under `isaaclab/external/m3po/` and launched from `isaaclab/scripts/reinforcement_learning/m3po/train.py`.

## Key Task IDs

- `Template-M4po-DataCollection-G1-InspireFTP-Abs-v0`
- `Isaac-PickPlace-G1-InspireFTP-Abs-v0`
- `Isaac-PickPlace-G1-InspireFTP-GR00T-Abs-v0`
- `M4PO-Inference-G1-InspireFTP-GR00T-Abs-v0`

## Installation

### Prerequisites

- Ubuntu Linux
- NVIDIA driver and a CUDA-capable GPU supported by Isaac Sim
- Conda
- `git-lfs`
- `unzip`

### 1. Clone the workspace

```bash
git clone --recurse-submodules <your-m4po-repo-url>
cd M4PO
git submodule update --init --recursive
```

### 2. Fetch the Unitree assets

```bash
cd m4po_datacollection
./fetch_assets.sh
cd ..
```

This populates `m4po_datacollection/assets/`, which is used by the local task and inference scripts.

### 3. Create the Isaac Lab environment

```bash
cd isaaclab
./isaaclab.sh --conda m4po_isaaclab
./isaaclab.sh --install none
cd ..
```

### 4. Activate the environment and install the local packages

```bash
source /home/aditya/miniconda3/etc/profile.d/conda.sh
conda activate m4po_isaaclab
python -m pip install --no-build-isolation -e isaaclab/source/isaaclab
python -m pip install --no-build-isolation -e isaaclab/source/isaaclab_tasks
python -m pip install --no-build-isolation -e isaaclab/source/isaaclab_assets
python -m pip install --no-build-isolation -e m4po_datacollection/source/m4po_datacollection
python -m pip install -e isaaclab/source/isaaclab_rl[m3po]
python -m pip install pyzmq msgpack msgpack-numpy
python -m pip install --no-deps --ignore-requires-python -e isaaclab/external/Isaac-GR00T
```

## Common Workflows

### Data collection and replay

```bash
source /home/aditya/miniconda3/etc/profile.d/conda.sh
conda activate m4po_isaaclab
cd m4po_datacollection
./scripts/list_envs.py --keyword DataCollection
./scripts/replay_dataset.py --device cpu --headless
```

### GR00T-style inference

```bash
source /home/aditya/miniconda3/etc/profile.d/conda.sh
conda activate m4po_isaaclab
python m4po-inference/download_model.py
bash m4po-inference/run_gr00t_server.sh
cd isaaclab
./isaaclab.sh -p ../m4po-inference/play.py \
  --server_host 127.0.0.1 \
  --server_port 5555 \
  --device cuda:0
```

### M4PO training

```bash
source /home/aditya/miniconda3/etc/profile.d/conda.sh
conda activate m4po_isaaclab
cd isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/m3po/train.py \
  --task Template-M4po-DataCollection-G1-InspireFTP-Abs-v0 \
  --num_envs 32 \
  --headless \
  steps=200000 \
  batch_size=256
```

## Important Paths

- Data-collection task package: `m4po_datacollection/source/m4po_datacollection/`
- Inference entry point: `m4po-inference/play.py`
- GR00T-style Isaac Lab task config: `isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_unitree_g1_inspire_hand_groot_env_cfg.py`
- M4PO launcher: `isaaclab/scripts/reinforcement_learning/m3po/train.py`
- Modified NEWT backend: `isaaclab/external/m3po/`

## Notes

- Large runtime artifacts such as datasets, downloaded checkpoints, logs, and Unitree USD assets are intentionally kept out of git.
- The GR00T-compatible pieces are the interface layer for pretraining and inference; M4PO is the main algorithmic contribution.
- The public GR00T checkpoints are not embodiment-matched to the Inspire hand, so the inference path uses local adapters and should be treated as a compatibility path until a task-matched checkpoint is trained.
