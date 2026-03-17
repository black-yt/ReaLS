#!/bin/bash
# SDE sampling from trained SiT model.
#
# Usage:
#   bash scripts/sample_sde.sh

set -e

# === Configuration ===
CKPT="/path/to/sit_checkpoint.pt"
VAE_PATH="/path/to/reals_vae"
MODEL="SiT-B/2"
GPUS_PER_NODE=4

# === Launch ===
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    generation/sample_ddp.py SDE \
    --model $MODEL \
    --ckpt $CKPT \
    --vae-path $VAE_PATH \
    --cfg-scale 1.5 \
    --num-sampling-steps 250 \
    --per-proc-batch-size 64 \
    --num-fid-samples 50000
