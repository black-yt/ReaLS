#!/bin/bash
# ReaLS alignment training launch script.
# Adjust --data-path, --vae-ckpt, and GPU settings as needed.
#
# Usage (single node, 4 GPUs):
#   bash scripts/train_alignment.sh
#
# Usage (multi-node with SLURM):
#   Modify NNODES and GPU_per_node, then:
#   srun ... torchrun ... alignment/train.py ...

set -e

# === Configuration ===
DATA_PATH="/path/to/imagenet"          # ImageNet root (containing train/ and val/)
VAE_CKPT=""                            # Optional: path to pretrained VAE state_dict
DINO_MODEL="dinov2_vitl14_reg"         # DINOv2 variant
OUTPUT_DIR="checkpoints"
GPUS_PER_NODE=4
NODES=1
BATCH_SIZE=8
LR=5e-5
MAX_EPOCH=10

# === Launch ===
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NODES \
    alignment/train.py \
    --data-path $DATA_PATH \
    --gpus-per-node $GPUS_PER_NODE \
    --nodes $NODES \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --max-epoch $MAX_EPOCH \
    --dino-model $DINO_MODEL \
    --output-dir $OUTPUT_DIR \
    ${VAE_CKPT:+--vae-ckpt $VAE_CKPT}
