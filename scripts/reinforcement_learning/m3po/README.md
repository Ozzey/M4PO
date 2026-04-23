# M3PO for Isaac Lab

This launcher wires the vendored M3PO codebase in `isaaclab/external/m3po` into Isaac Lab tasks, including the local M4PO data-collection task package.

## Install the Isaac Lab-side extras

```bash
source /home/aditya/miniconda3/etc/profile.d/conda.sh
conda activate m4po_isaaclab
python -m pip install -e source/isaaclab_rl[m3po]
```

## Train on the local G1 Inspire pick-place task

```bash
./isaaclab.sh -p scripts/reinforcement_learning/m3po/train.py \
  --task Template-M4po-DataCollection-G1-InspireFTP-Abs-v0 \
  --num_envs 32 \
  --headless \
  steps=200000 \
  batch_size=256
```

## Notes

- The current Isaac Lab backend supports `obs=state`.
- You can pass normal M3PO Hydra overrides after the launcher arguments, for example `model_size=B`, `steps=1000000`, or `enable_wandb=true`.
- Task metadata for Isaac Lab tasks is probed dynamically from the registered gym environment, so you do not need to add them to `tasks.json` for single-task runs.
