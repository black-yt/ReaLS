"""
Main training script for ReaLS alignment (VAE + DINOv2).
Uses PyTorch Lightning with DDP for multi-GPU training.
"""

import os
import datetime
import argparse

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from alignment.models.vae_dino import VaeDino
from alignment.lit_model import AlignModel


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
    """
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


def train_model(args):
    save_path = os.path.join(
        args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    )

    torch_model = VaeDino(
        vae_ckpt=args.vae_ckpt,
        dino_model=args.dino_model,
        align_cls=args.align_cls,
        img_size=args.image_size,
        dino_input_size=args.dino_input_size,
        vae_loss_weight=args.vae_loss_weight,
        disc_ckpt=args.disc_ckpt,
    )

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    train_data = datasets.ImageFolder(
        root=os.path.join(args.data_path, "train"), transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_data = datasets.ImageFolder(
        root=os.path.join(args.data_path, "val"), transform=transform
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    world_size = args.gpus_per_node * args.nodes
    steps_per_epoch = len(train_loader) // world_size

    lit_model = AlignModel(
        torch_model,
        lr=args.lr,
        eta_min=args.eta_min,
        max_epoch=args.max_epoch,
        steps_per_epoch=steps_per_epoch,
        visualization_folder=os.path.join(save_path, "visualization"),
        use_ema=args.use_ema,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
        filename="{epoch:02d}-{val_loss:.4f}",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience, check_finite=True
    )

    trainer = L.Trainer(
        default_root_dir="./",
        log_every_n_steps=50,
        precision="16-mixed",
        max_epochs=args.max_epoch,
        accelerator="gpu",
        devices=args.gpus_per_node,
        num_nodes=args.nodes,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.print("[checkpoint path]", save_path)
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReaLS alignment training")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ImageNet root (containing train/ and val/ subdirs)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--vae-ckpt", type=str, default=None,
                        help="Path to pretrained VAE state_dict")
    parser.add_argument("--dino-model", type=str, default="dinov2_vitl14_reg",
                        choices=list(VaeDino.DINO_DIMS.keys()),
                        help="DINOv2 model variant")
    parser.add_argument("--dino-input-size", type=int, default=448)
    parser.add_argument("--align-cls", action="store_true", default=True,
                        help="Also align CLS token")
    parser.add_argument("--no-align-cls", action="store_false", dest="align_cls")
    parser.add_argument("--vae-loss-weight", type=float, default=1.0)
    parser.add_argument("--disc-ckpt", type=str, default=None,
                        help="Path to pretrained discriminator checkpoint")

    # Training
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", action="store_false", dest="use_ema")

    # Distributed
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train_model(args)
