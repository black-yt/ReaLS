"""
VaeDino: VAE encoder/decoder aligned with DINOv2 features.

The VAE latents are projected to match DINOv2 patch/class tokens via learned
alignment networks (MLPs), producing a representation-aligned latent space.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from diffusers import AutoencoderKL

from .losses import LPIPSWithDiscriminator


def mse_loss(x, y):
    return ((x - y) ** 2).mean()


def mae_loss(x, y):
    return (torch.abs(x - y)).mean()


class VaeDino(nn.Module):
    """
    VAE + DINOv2 alignment model.

    Trains the VAE so that its latent space is aligned with DINOv2 features,
    producing semantically meaningful latent representations.

    Args:
        vae_ckpt: path to pretrained VAE state_dict (None to init from scratch).
        dino_model: DINOv2 model name for torch.hub, e.g. "dinov2_vitl14_reg".
        align_cls: whether to also align the CLS token.
        img_size: input image resolution (must be divisible by 8).
        dino_input_size: resolution for DINOv2 input (must be divisible by 14).
        vae_loss_weight: weight for the VAE reconstruction/GAN loss.
        disc_ckpt: optional path to a pretrained discriminator checkpoint.
    """

    DINO_DIMS = {
        "dinov2_vitb14": 768,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14": 1536,
        "dinov2_vitg14_reg": 1536,
    }

    def __init__(
        self,
        vae_ckpt=None,
        dino_model="dinov2_vitl14_reg",
        align_cls=True,
        img_size=256,
        dino_input_size=448,
        vae_loss_weight=1.0,
        disc_ckpt=None,
    ):
        super().__init__()

        # --- VAE ---
        self.vae = AutoencoderKL(
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
        if vae_ckpt is not None:
            checkpoint = torch.load(vae_ckpt, map_location="cpu")
            self.vae.load_state_dict(checkpoint)
            print(f"[loaded VAE from {vae_ckpt}]")

        # --- DINOv2 (frozen) ---
        assert dino_model in self.DINO_DIMS, f"Unsupported DINO model: {dino_model}"
        dino_out_dim = self.DINO_DIMS[dino_model]
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_model)
        self._freeze(self.dino)
        self.dino.eval()
        print(f"[loaded DINOv2: {dino_model}, dim={dino_out_dim}]")

        # --- Alignment geometry ---
        assert img_size % 8 == 0
        assert dino_input_size % 14 == 0
        assert (img_size // 8) % (dino_input_size // 14) == 0
        self.img_size = img_size
        self.dino_input_size = dino_input_size
        self.vae_z_merge = (img_size // 8) // (dino_input_size // 14)
        print(f"[vae_z_merge {self.vae_z_merge}]")

        # --- Alignment networks ---
        align_in_dim = 4 * (self.vae_z_merge ** 2)
        self.align_net = nn.Sequential(
            nn.LayerNorm(align_in_dim) if self.vae_z_merge > 1 else nn.Identity(),
            nn.Linear(align_in_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, dino_out_dim),
        )

        self.align_cls = align_cls
        if align_cls:
            self.align_cls_net = nn.Sequential(
                nn.LayerNorm(align_in_dim) if self.vae_z_merge > 1 else nn.Identity(),
                nn.Linear(align_in_dim, 1024),
                nn.GELU(),
                nn.Linear(1024, dino_out_dim),
            )
            print("[align cls]")

        # --- Loss modules ---
        self.vae_loss_model = LPIPSWithDiscriminator(disc_ckpt=disc_ckpt)
        self.vae_loss_weight = vae_loss_weight

        self.cos_loss_fun = nn.CosineEmbeddingLoss()
        self.l1_smooth_fun = nn.SmoothL1Loss()

        self.IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        print(f"[ae params: {sum(p.numel() for p in self.ae_parameters() if p.requires_grad)}]")
        print(f"[disc params: {sum(p.numel() for p in self.disc_parameters() if p.requires_grad)}]")

    def ae_parameters(self):
        """Parameters for the autoencoder + alignment networks (generator side).

        Note: align_cls_net is intentionally excluded. It participates in loss
        computation (providing gradients to the VAE), but its own weights are
        not updated by the optimizer. This matches the original training setup
        that produced the published results.
        """
        return list(self.vae.parameters()) + list(self.align_net.parameters())

    def disc_parameters(self):
        """Parameters for the discriminator."""
        return self.vae_loss_model.parameters()

    @staticmethod
    def _freeze(module):
        for p in module.parameters():
            p.requires_grad = False

    def cos_loss(self, x, y):
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        target = torch.ones_like(x[:, 0])
        return self.cos_loss_fun(x, y, target)

    def l1_smooth(self, x, y):
        return self.l1_smooth_fun(x, y)

    def dino_transform(self, x):
        """Transform VAE-range images ([-1, 1]) to ImageNet-normalized images."""
        x = x / 2 + 0.5
        x = (x - self.IMAGENET_DEFAULT_MEAN.to(x.device)) / self.IMAGENET_DEFAULT_STD.to(x.device)
        return x

    def align_loss(self, vae_latents, dino_feature):
        """Compute alignment loss between VAE latents and DINOv2 patch tokens."""
        B, C, H, W = vae_latents.shape
        vae_latents = vae_latents.permute(0, 2, 3, 1).contiguous()
        vae_latents = vae_latents.reshape(B, H * W, C).contiguous()
        vae_feature = self.align_net(vae_latents)
        return 0.9 * self.cos_loss(vae_feature, dino_feature) + 0.1 * self.l1_smooth(vae_feature, dino_feature)

    def align_cls_loss(self, vae_latents, dino_cls_feature):
        """Compute alignment loss between VAE global feature and DINOv2 CLS token."""
        vae_latents = vae_latents.mean(dim=[-1, -2]).contiguous()  # B, 4
        vae_cls_feature = self.align_cls_net(vae_latents)
        return 0.9 * self.cos_loss(vae_cls_feature, dino_cls_feature) + 0.1 * self.l1_smooth(vae_cls_feature, dino_cls_feature)

    def forward(self, inputs, optimizer_idx):
        """
        Forward pass with VAE encoding/decoding and alignment loss computation.

        Args:
            inputs: images in [-1, 1], shape (B, 3, H, W).
            optimizer_idx: 0 for generator step, 1 for discriminator step.
        """
        assert inputs.shape[-2] == self.img_size and inputs.shape[-1] == self.img_size
        posterior = self.vae.encode(inputs).latent_dist
        z = posterior.sample()
        reconstructions = self.vae.decode(z).sample

        output_dict = {
            "posterior": posterior,
            "reconstructions": reconstructions,
        }

        if optimizer_idx == 0:
            vae_loss, loss_log = self.vae_loss_model(inputs, reconstructions, posterior, 0)

            with torch.no_grad():
                inputs_dino = F.interpolate(
                    inputs, size=(self.dino_input_size, self.dino_input_size),
                    mode="bilinear", align_corners=False,
                )
                inputs_dino = self.dino_transform(inputs_dino)
                dino_feature = self.dino.forward_features(inputs_dino)

            if self.vae_z_merge > 1:
                z = rearrange(
                    z, "b c (num_h win_h) (num_w win_w) -> b (c win_h win_w) num_h num_w",
                    win_h=self.vae_z_merge, win_w=self.vae_z_merge,
                )

            align_loss = self.align_loss(z, dino_feature["x_norm_patchtokens"])
            loss_log["align_loss"] = align_loss.item()

            if self.align_cls:
                cls_loss = self.align_cls_loss(z, dino_feature["x_norm_clstoken"])
                loss_log["cls_loss"] = cls_loss.item()
                total_loss = 0.5 * (align_loss + cls_loss) + self.vae_loss_weight * vae_loss
            else:
                total_loss = align_loss + self.vae_loss_weight * vae_loss

            loss_log["total_loss"] = total_loss.item()
            return total_loss, loss_log, output_dict

        elif optimizer_idx == 1:
            disc_loss, loss_log = self.vae_loss_model(inputs, reconstructions, posterior, 1)
            return disc_loss, loss_log, output_dict
