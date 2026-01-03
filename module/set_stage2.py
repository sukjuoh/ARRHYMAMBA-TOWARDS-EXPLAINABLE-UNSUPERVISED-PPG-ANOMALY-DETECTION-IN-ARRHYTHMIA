"""
ref: https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection/blob/main/experiments/exp_stage2.py
"""


import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from models.stage2.ArrhyMamba import ArrhyMamba


from utils import linear_warmup_cosine_annealingLR


class SetStage2(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = ArrhyMamba(**config['ArrhyMamba'], config=config)

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def training_step(self, batch, batch_idx):
        self.train()
        x = batch

        mask_pred_loss = self.model(x)
        loss = mask_pred_loss

        sch = self.lr_schedulers()
        sch.step()

        loss_hist = {'loss': loss, 'mask_pred_loss': mask_pred_loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()  # set the model on the evaluation mode
        x = batch

        mask_pred_loss = self.model(x)
        loss = mask_pred_loss

    
        loss_hist = {'loss': loss, 'mask_pred_loss': mask_pred_loss}
        for k, v in loss_hist.items():
            if v is not None: 
                self.log(f'val/{k}', v)

        if batch_idx == 0:
            self.model.eval()

            # unconditional sampling
            s = self.model.iterative_decoding(device=self.device)
            xhat = self.model.decode_token_ind_to_timeseries(s).cpu()

            b = 0
            fig, axes = plt.subplots(2, 1, figsize=(4, 1.5 * 2))
            plt.suptitle(f'step-{self.global_step}')
            axes[0].set_title(r'$\hat{x}$')
            axes[0].plot(xhat[b, 0, :])
            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.tight_layout()
            self.logger.log_image(
                key=f"generated sample ({'train' if self.training else 'val'})",
                images=[wandb.Image(plt)]
            )
            plt.close()

        return loss_hist
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['exp_params']['lr']['stage2'],
            betas=(0.9, 0.95),
            weight_decay=5 * 1e-2
        )

        scheduler = linear_warmup_cosine_annealingLR(
            opt,
            self.config['trainer_params']['max_steps']['stage2'],
            self.config['exp_params']['linear_warmup_rate']
        )

        return {'optimizer': opt, 'lr_scheduler': scheduler}

