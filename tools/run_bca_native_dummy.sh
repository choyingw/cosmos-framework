#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
# Fresh 10-step native-mask student, export, then ONE profiled H200 inference.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
: "${EDGE_HF:?Set EDGE_HF to the previously fine-tuned dense edge teacher export}"
run_dir=${BCA_NATIVE_RUN:-/workspace/cosmos_data/bca_native_run01}
python_bin=${BCA_PYTHON:-"$repo_root/.venv/bin/python"}
gpus=${BCA_GPUS:-8}
if [[ -e "$run_dir" ]]; then
    echo "Use a fresh BCA_NATIVE_RUN; already exists: $run_dir" >&2
    exit 1
fi
# Keep this run independent of diagnostic overrides from an earlier shell.
unset COSMOS_BCA_NATIVE_CROSS_VIEW COSMOS_BCA_BACKEND
export COSMOS_INTERNAL=0
# Fail before training if this H200 environment cannot load the requested library.
"$python_bin" -c 'import torch; from cosmos_framework.model.generator.mot.bca_attention import _inference_flash_backend; _inference_flash_backend(torch.device("cuda:0"), "flash3")'
COSMOS_TRAINING=1 "$python_bin" -m cosmos_framework.scripts.prepare_transfer_auto_edge_sample \
    --train-dummy --bca --steps 10 --gpus "$gpus" \
    --checkpoint "$EDGE_HF" --output "$run_dir"
source "$run_dir/train.env"
train_dir="$run_dir/training/cosmos3/transfer_auto/bca_dummy"
export_dir="$run_dir/bca_hf"
COSMOS_TRAINING=1 "$python_bin" -m cosmos_framework.scripts.export_model \
    --config-file "$train_dir/config.yaml" \
    --checkpoint-path "$train_dir/checkpoints/iter_000000010" \
    --output-dir "$export_dir" --no-vit --bundle-lidar-vae \
    --per-view-captions on --per-view-caption-task av_multiview_edge_transfer
COSMOS_TRAINING=0 COSMOS_BCA_FLASH_BACKEND=flash3 "$python_bin" -m cosmos_framework.scripts.run_transfer_auto \
    --checkpoint "$export_dir" --gpus "$gpus" \
    --control-type edge --mode camera --frames 61 --resolution 480 --seed 0 \
    --warmup 0 --num-iterations 1 --profile-steps --profile-step-attention \
    --output "$run_dir/inference"
