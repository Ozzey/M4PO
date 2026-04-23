#!/usr/bin/env bash

set -euo pipefail

REPO_NAME="unitree_sim_isaaclab_usds"
TARGET_DIR="assets"

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs is required. Install it first, then rerun this script."
    exit 1
fi

rm -rf "${TARGET_DIR}" "${REPO_NAME}"

git lfs install
git clone "https://huggingface.co/datasets/unitreerobotics/${REPO_NAME}"

cd "${REPO_NAME}"

if [ ! -f "assets.zip" ]; then
    echo "assets.zip is missing after clone."
    exit 1
fi

filesize=$(stat -c%s "assets.zip")
if [ "${filesize}" -le $((1024 * 1024 * 1024)) ]; then
    echo "assets.zip looks incomplete (${filesize} bytes). Check Git LFS access."
    exit 1
fi

unzip -q assets.zip

if [ ! -d "assets" ]; then
    echo "The Unitree asset archive did not unpack an assets directory."
    exit 1
fi

mv assets ../
cd ..
rm -rf "${REPO_NAME}"

echo "Assets downloaded into $(pwd)/assets"
