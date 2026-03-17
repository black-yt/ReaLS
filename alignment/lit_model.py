"""
PyTorch Lightning module for ReaLS alignment training.
Handles dual-optimizer (AE + discriminator), EMA, visualization, and LR scheduling.
"""

import os
import torch
import lightning as L
from torch import nn
from torchvision.utils import save_image
from copy import deepcopy
from collections import OrderedDict


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


class EMA(nn.Module):
    """Exponential Moving Average of model parameters."""

    def __init__(self, model):
        super().__init__()
        self.ema_model = deepcopy(model)
        requires_grad(self.ema_model, False)
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, model, decay=0.9999):
        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class AlignModel(L.LightningModule):
    """
    Lightning module wrapping VaeDino for alignment training.

    Implements manual optimization with two optimizers (generator + discriminator),
    cosine annealing LR schedule, and optional EMA.

    Args:
        model: a VaeDino instance.
        lr: learning rate.
        eta_min: minimum LR for cosine annealing.
        max_epoch: total training epochs.
        steps_per_epoch: steps per epoch (for LR schedule).
        visualization_folder: path to save reconstruction visualizations (None to disable).
        visualization_freq: save visualizations every N steps.
        use_ema: whether to maintain an EMA copy of the model.
    """

    def __init__(
        self,
        model: nn.Module = nn.Identity(),
        lr=0.0001,
        eta_min=0.0,
        max_epoch=10,
        steps_per_epoch=100,
        visualization_folder=None,
        visualization_freq=1000,
        use_ema=True,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.eta_min = eta_min
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = (steps_per_epoch + 1) * max_epoch
        self.visualization_folder = visualization_folder
        if self.visualization_folder is not None:
            os.makedirs(os.path.join(self.visualization_folder, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.visualization_folder, "val"), exist_ok=True)
        self.visualization_freq = visualization_freq
        self.global_train_step = 0
        self.global_val_step = 0
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(self.model)
        self.automatic_optimization = False
        self.example_input_array = torch.Tensor(4, 3, 256, 256)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        x = self.model(x, 0)
        return x

    def training_step(self, batch):
        self.global_train_step += 1

        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()
        x, y = batch

        # Generator step
        g_loss, g_loss_dict, output_dict = self.model(x, 0)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        g_sch.step()

        if (
            self.visualization_folder is not None
            and self.global_train_step % self.visualization_freq == 0
            and self.global_rank == 0
        ):
            visual_output = torch.cat([x, output_dict["reconstructions"]], dim=0)
            visual_output = torch.clip((visual_output + 1) * 0.5, 0, 1)
            save_image(
                visual_output,
                os.path.join(self.visualization_folder, "train", f"{self.global_train_step:08}.jpeg"),
                nrow=int(visual_output.shape[0] // 2),
                padding=0,
            )

        # Discriminator step
        d_loss, d_loss_dict, _ = self.model(x, 1)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        d_sch.step()

        if self.use_ema:
            self.ema.update(self.model)

        lr_now = self._g_optimizer.param_groups[0]["lr"]
        log_dict = {"lr": lr_now, "train_loss": g_loss}
        g_loss_dict.pop("total_loss")
        log_dict.update(g_loss_dict)
        log_dict.update(d_loss_dict)

        log_dict_short = {}
        for k, v in log_dict.items():
            log_dict_short[k.replace("_loss", "")] = v
        assert len(log_dict_short) == len(log_dict)

        self.log_dict(log_dict_short, prog_bar=True)

    def validation_step(self, batch):
        self.global_val_step += 1

        x, y = batch
        g_loss, g_loss_dict, output_dict = self.model(x, 0)
        d_loss, d_loss_dict, _ = self.model(x, 1)

        if (
            self.visualization_folder is not None
            and self.global_val_step % self.visualization_freq == 0
            and self.global_rank == 0
        ):
            visual_output = torch.cat([x, output_dict["reconstructions"]], dim=0)
            visual_output = torch.clip((visual_output + 1) * 0.5, 0, 1)
            save_image(
                visual_output,
                os.path.join(self.visualization_folder, "val", f"{self.global_val_step:08}.jpeg"),
                nrow=int(visual_output.shape[0] // 2),
                padding=0,
            )

        log_dict = {"val_loss": g_loss, "val_disc_loss": d_loss}
        self.log_dict(log_dict, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.ae_parameters(), lr=self.lr, weight_decay=0.0, betas=(0.5, 0.9)
        )
        self._g_optimizer = optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.total_steps, eta_min=self.eta_min
        )
        scheduler_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        disc_optimizer = torch.optim.AdamW(
            self.model.disc_parameters(), lr=self.lr, weight_decay=0.0, betas=(0.5, 0.9)
        )
        disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            disc_optimizer, T_max=self.total_steps, eta_min=self.eta_min
        )
        disc_scheduler_dict = {"scheduler": disc_scheduler, "interval": "step", "frequency": 1}

        return [optimizer, disc_optimizer], [scheduler_dict, disc_scheduler_dict]
