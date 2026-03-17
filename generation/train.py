# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json

from generation.models import SiT_models
from generation.transport import create_transport, Sampler
from generation.train_utils import parse_transport_args
from diffusers.models import AutoencoderKL

SCALE_FACTOR = 0.18215

# Latent normalization statistics for the original SD-VAE
OLD_MEAN = 0.292873
OLD_STD = 68.317589 - (-65.730583)  # max - min


def new2old(z, new_mean, new_std):
    """Convert from ReaLS latent space to original SD-VAE latent space."""
    z = (z - new_mean) / new_std
    z = z * OLD_STD + OLD_MEAN
    return z


def old2new(z, new_mean, new_std):
    """Convert from original SD-VAE latent space to ReaLS latent space."""
    z = (z - OLD_MEAN) / OLD_STD
    z = z * new_std + new_mean
    return z


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from ADM."""
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
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """Trains a new SiT model."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                          f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        visualization_dir = f"{experiment_dir}/visualization"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    print(f"[diffusion model] {args.model}")

    ema = deepcopy(model).to(device)

    # Load checkpoint (if resuming)
    ckpt_data = None
    if args.ckpt is not None:
        ckpt_data = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt_data["model"])
        ema.load_state_dict(ckpt_data["ema"])

    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])

    transport = create_transport(
        args.path_type, args.prediction, args.loss_weight,
        args.train_eps, args.sample_eps,
    )
    transport_sampler = Sampler(transport)

    # Load ReaLS VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    print(f"VAE path: {args.vae_path}")
    with open(os.path.join(args.vae_path, "mean_std.json"), "r", encoding="utf-8") as f:
        mean_std = json.load(f)
    new_mean = mean_std["mean"]
    new_std = mean_std["max"] - mean_std["min"]
    print(f"{OLD_MEAN = }, {OLD_STD = }, {new_mean = }, {new_std = }")

    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    if ckpt_data is not None and "opt" in ckpt_data:
        opt.load_state_dict(ckpt_data["opt"])

    # Data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank,
        shuffle=True, seed=args.global_seed,
    )
    loader = DataLoader(
        dataset, batch_size=local_batch_size, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare for training
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Monitoring variables
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels for periodic visualization
    ys = torch.randint(1000, size=(local_batch_size // 2,), device=device)
    use_cfg = args.cfg_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = new2old(vae.encode(x).latent_dist.sample(), new_mean, new_std).mul_(SCALE_FACTOR)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                sample_fn = transport_sampler.sample_ode()
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                dist.barrier()

                if use_cfg:
                    samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(old2new(samples / SCALE_FACTOR, new_mean, new_std)).sample
                out_samples = torch.zeros(
                    (args.global_batch_size // 2, 3, args.image_size, args.image_size), device=device,
                )
                dist.all_gather_into_tensor(out_samples, samples)
                if rank == 0:
                    out_samples = torch.clip((out_samples + 1) / 2, 0, 1)
                    save_image(out_samples, f"{visualization_dir}/{train_steps:07d}.png",
                               nrow=out_samples.shape[0] // 4, padding=0)
                logging.info("Generating EMA samples done.")

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ImageNet training set")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100_000)
    parser.add_argument("--sample-every", type=int, default=50_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--vae-path", type=str, required=True,
                        help="Path to ReaLS VAE (diffusers format, must contain mean_std.json)")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
