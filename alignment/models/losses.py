"""
LPIPS perceptual loss + PatchGAN discriminator for VAE training.
Self-contained: uses the `lpips` package instead of `taming`.
"""

import torch
import torch.nn as nn
import lpips


# ---------------------------------------------------------------------------
#  PatchGAN (NLayerDiscriminator)
#  Standard PatchGAN from pix2pix; classifies overlapping patches as real/fake.
# ---------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator from pix2pix (Isola et al., 2017)."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = False

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)


# ---------------------------------------------------------------------------
#  Discriminator loss functions
# ---------------------------------------------------------------------------

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.relu(1.0 - logits_real))
    loss_fake = torch.mean(torch.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.softplus(-logits_real))
    loss_fake = torch.mean(torch.nn.functional.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


# ---------------------------------------------------------------------------
#  LPIPSWithDiscriminator
# ---------------------------------------------------------------------------

class LPIPSWithDiscriminator(nn.Module):
    """
    Combined loss module: reconstruction (L1) + perceptual (LPIPS) + KL + GAN.

    Args:
        kl_weight: weight for the KL divergence term.
        disc_num_layers: number of layers in PatchGAN discriminator.
        disc_in_channels: input channels for discriminator.
        d_weight: weight for discriminator loss.
        g_weight: weight for generator (adversarial) loss.
        perceptual_weight: weight for LPIPS perceptual loss.
        disc_loss: type of discriminator loss ("hinge" or "vanilla").
        disc_ckpt: optional path to a pretrained discriminator checkpoint.
    """

    def __init__(
        self,
        kl_weight=2e-5,
        disc_num_layers=3,
        disc_in_channels=3,
        d_weight=1.0,
        g_weight=0.1,
        perceptual_weight=1.0,
        disc_loss="hinge",
        disc_ckpt=None,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
        self.perceptual_weight = perceptual_weight
        print(f"[kl_weight: {kl_weight}]")

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
        ).apply(weights_init)

        if disc_ckpt is not None:
            disc_state = torch.load(disc_ckpt, map_location="cpu")
            self.discriminator.load_state_dict(disc_state, strict=True)
            print(f"[discriminator loaded from {disc_ckpt}]")
        else:
            print("[discriminator weights init]")

        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.d_weight = d_weight
        self.g_weight = g_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean()
        p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()).mean()
        kl_loss = posteriors.kl().mean()

        if optimizer_idx == 0:
            # Generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            loss = rec_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss + self.g_weight * g_loss

            log = {
                "vae_loss": loss.clone().detach().mean().item(),
                "kl_loss": kl_loss.detach().mean().item(),
                "rec_loss": rec_loss.detach().mean().item(),
                "p_loss": p_loss.detach().mean().item(),
                "g_loss": g_loss.detach().mean().item(),
            }
            return loss, log

        if optimizer_idx == 1:
            # Discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            d_loss = self.d_weight * self.disc_loss(logits_real, logits_fake)

            log = {
                "disc_loss": d_loss.clone().detach().mean().item(),
                "logits_real": logits_real.detach().mean().item(),
                "logits_fake": logits_fake.detach().mean().item(),
            }
            return d_loss, log
