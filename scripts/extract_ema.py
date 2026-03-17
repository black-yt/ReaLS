"""
Extract EMA weights from a Lightning or SiT checkpoint for inference.

Usage:
    # From Lightning alignment checkpoint:
    python scripts/extract_ema.py --ckpt path/to/alignment.ckpt --output ema_model.pt --type alignment

    # From SiT training checkpoint:
    python scripts/extract_ema.py --ckpt path/to/sit.pt --output ema_model.pt --type sit
"""

import argparse
import torch


def extract_alignment_ema(ckpt_path, output_path):
    """Extract EMA weights from a Lightning alignment checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    ema_prefix = "ema.ema_model."
    ema_state = {}
    for k, v in state_dict.items():
        if k.startswith(ema_prefix):
            new_key = k[len(ema_prefix):]
            ema_state[new_key] = v

    if not ema_state:
        print("No EMA weights found. Extracting model weights instead.")
        model_prefix = "model."
        for k, v in state_dict.items():
            if k.startswith(model_prefix) and not k.startswith("model.vae_loss_model."):
                new_key = k[len(model_prefix):]
                ema_state[new_key] = v

    torch.save(ema_state, output_path)
    print(f"Saved EMA weights to {output_path} ({len(ema_state)} keys)")


def extract_sit_ema(ckpt_path, output_path):
    """Extract EMA weights from a SiT training checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "ema" in checkpoint:
        ema_state = checkpoint["ema"]
    else:
        print("No 'ema' key found in checkpoint, using full checkpoint as state_dict.")
        ema_state = checkpoint

    torch.save(ema_state, output_path)
    print(f"Saved EMA weights to {output_path} ({len(ema_state)} keys)")


def main():
    parser = argparse.ArgumentParser(description="Extract EMA weights from checkpoints")
    parser.add_argument("--ckpt", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output path for EMA weights")
    parser.add_argument("--type", type=str, required=True, choices=["alignment", "sit"],
                        help="Checkpoint type")
    args = parser.parse_args()

    if args.type == "alignment":
        extract_alignment_ema(args.ckpt, args.output)
    elif args.type == "sit":
        extract_sit_ema(args.ckpt, args.output)


if __name__ == "__main__":
    main()
