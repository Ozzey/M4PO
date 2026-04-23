# m4po-inference

Local GR00T inference toolkit for the M4PO G1 Inspire 5-finger pick-and-place task.

What is here:
- A local-assets Isaac Lab env registration for G1 + Inspire 5-finger pick-place.
- A GR00T play script that adapts the public Arena G1 action head to the Inspire FTP action space.
- Helpers to download a GR00T checkpoint into this folder and run a policy server.

Env ID:
- `M4PO-Inference-G1-InspireFTP-GR00T-Abs-v0`

Quick start:

```bash
python m4po-inference/download_model.py
```

```bash
bash m4po-inference/run_gr00t_server.sh
```

```bash
./isaaclab.sh -p m4po-inference/play.py \
  --server_host 127.0.0.1 \
  --server_port 5555 \
  --device cuda:0
```

Notes:
- The env uses the local workspace robot and table USDs from `m4po_datacollection/assets`.
- The default public checkpoint is `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`.
- That checkpoint is trained for Arena G1 loco-manipulation, not this exact fixed-base Inspire embodiment, so the play script uses a compatibility adapter. It is useful for smoke tests and experimentation, but a finetuned checkpoint for this exact embodiment will be better.
