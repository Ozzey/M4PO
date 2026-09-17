# Cluster runs

Cluster manifests are deployment-specific and are not checked in. Use
`scripts/train_m4po.sh` as the container entry point and persist the configured
`log_dir`, which contains resolved configuration, JSONL metrics, and strict
checkpoints.

