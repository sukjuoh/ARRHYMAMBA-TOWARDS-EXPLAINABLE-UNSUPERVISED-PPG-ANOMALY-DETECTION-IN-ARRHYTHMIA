import os
from argparse import ArgumentParser
import datetime

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
import torch.nn.init as init
from pytorch_lightning.callbacks import EarlyStopping


from exp.set_stage1 import SetStage1

from preprocessing.preprocess import PPG_NormalSequence
from preprocessing.data_pipeline import build_pretrain_data_pipeline
from utils import get_root_dir


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int)
    parser.add_argument('--input_shape', default=200, type=int)
    return parser.parse_args()



def train_stage1(config: dict,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind: list,
                 checkpoint_path: str = None,
                ) -> pl.Trainer:
    """
    Train the Stage 1 model. Continues training from a checkpoint if provided.
    """
    project_name = 'ArrhyMamba_Stage1(VQ-VAE)'
    train_exp = SetStage1(config)



    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    extra_config = {'n_trainable_params': n_trainable_params, 'gpu_device_ind': gpu_device_ind}
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, **extra_config})


    early_stopping_callback = EarlyStopping(
        monitor='val/loss',  
        patience=100,          
        verbose=True,
        mode='min',           
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval='step'), early_stopping_callback] if early_stopping_callback else [LearningRateMonitor(logging_interval='step')],
        max_steps=config['trainer_params']['max_steps']['stage1'],
        devices=gpu_device_ind,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true' if len(gpu_device_ind) > 1 else "auto",
        val_check_interval=config['trainer_params']['val_check_interval']['stage1'],
        check_val_every_n_epoch=None,
        max_time=datetime.timedelta(hours=config['trainer_params']['max_hours']['stage1']),
    )
    '''
     callbacks=[LearningRateMonitor(logging_interval='step')],
     '''
    
    trainer.fit(
        train_exp,
        train_dataloaders=train_data_loader,
        val_dataloaders=test_data_loader,  
    )
    
    return trainer
