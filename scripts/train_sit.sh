#!/bin/bash
# SiT training in ReaLS latent space.
#
# Usage (single node, 4 GPUs):
#   bash scripts/train_sit.sh

set -e

# === Configuration ===
DATA_PATH="/path/to/imagenet/train"    # ImageNet training set
VAE_PATH="/path/to/reals_vae"         # ReaLS VAE (diffusers format, must contain mean_std.json)
MODEL="SiT-B/2"
GPUS_PER_NODE=4
NODES=1
GLOBAL_BATCH=$((GPUS_PER_NODE * NODES * 32))

# === Launch ===
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NODES \
    generation/train.py \
    --data-path $DATA_PATH \
    --vae-path $VAE_PATH \
    --model $MODEL \
    --global-batch-size $GLOBAL_BATCH \
    --epochs 400 \
    --log-every 100 \
    --ckpt-every 100000 \
    --sample-every 50000 \
    --cfg-scale 4.0
