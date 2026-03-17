"""
Convert a trained ReaLS alignment checkpoint to diffusers AutoencoderKL format.

This script:
1. Loads the alignment Lightning checkpoint.
2. Extracts the VAE (EMA) weights.
3. Saves them in diffusers `AutoencoderKL.from_pretrained()` format.
4. Computes and saves latent normalization statistics (mean_std.json).

Usage:
    python scripts/convert_to_diffusers.py \
        --ckpt path/to/alignment.ckpt \
        --output-dir path/to/vae_diffusers/ \
        --data-path /path/to/imagenet/train \
        --num-samples 10000
"""

import argparse
import json
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def extract_vae_state_dict(ckpt_path, use_ema=True):
    """Extract VAE state_dict from a Lightning alignment checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    if use_ema:
        prefix = "ema.ema_model.vae."
    else:
        prefix = "model.vae."

    vae_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            vae_state[new_key] = v

    if not vae_state:
        raise ValueError(f"No VAE weights found with prefix '{prefix}' in checkpoint.")

    return vae_state


def compute_latent_stats(vae, data_path, image_size=256, num_samples=10000, batch_size=64):
    """Compute latent space normalization statistics."""
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = next(vae.parameters()).device
    all_latents = []
    count = 0

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Computing latent stats"):
            if count >= num_samples:
                break
            x = x.to(device)
            z = vae.encode(x).latent_dist.sample()
            all_latents.append(z.cpu())
            count += x.shape[0]

    all_latents = torch.cat(all_latents, dim=0)[:num_samples]
    mean_val = all_latents.mean().item()
    min_val = all_latents.min().item()
    max_val = all_latents.max().item()

    return {"mean": mean_val, "min": min_val, "max": max_val}


def main():
    parser = argparse.ArgumentParser(description="Convert alignment checkpoint to diffusers format")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning alignment checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for diffusers model")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to ImageNet training set (for computing mean_std.json)")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples for computing latent statistics")
    parser.add_argument("--no-ema", action="store_true", help="Use model weights instead of EMA")
    parser.add_argument("--mean-std-json", type=str, default=None,
                        help="Provide pre-computed mean_std.json instead of computing from data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract VAE weights
    print("Extracting VAE weights from checkpoint...")
    vae_state = extract_vae_state_dict(args.ckpt, use_ema=not args.no_ema)

    # Create and load VAE
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=32,
        sample_size=256,
    )
    vae.load_state_dict(vae_state)
    print(f"Loaded VAE with {sum(p.numel() for p in vae.parameters()):,} parameters")

    # Save in diffusers format
    vae.save_pretrained(args.output_dir)
    print(f"Saved diffusers model to {args.output_dir}")

    # Handle mean_std.json
    if args.mean_std_json is not None:
        # Copy provided mean_std.json
        import shutil
        shutil.copy(args.mean_std_json, os.path.join(args.output_dir, "mean_std.json"))
        print(f"Copied mean_std.json from {args.mean_std_json}")
    elif args.data_path is not None:
        # Compute from data
        print("Computing latent normalization statistics...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = vae.to(device).eval()
        stats = compute_latent_stats(vae, args.data_path, num_samples=args.num_samples)
        stats_path = os.path.join(args.output_dir, "mean_std.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved latent statistics to {stats_path}: {stats}")
    else:
        print("Warning: No --data-path or --mean-std-json provided. "
              "mean_std.json is required for SiT training. "
              "Please compute it separately or provide it later.")


if __name__ == "__main__":
    main()
