#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl

import albumentations as A
from torch.utils.data import DataLoader
from dataset import PANDADataset

class MyCallback(pl.Callback):
    def on_epoch_start(self, trainer, pl_module):
        if pl_module.hparams.progressive:
            ind = int(trainer.current_epoch / 5) if trainer.current_epoch < 5*7 else 7
            prog = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
            batch = [32, 32, 32, 16, 8, 8, 4, 4]
            # For Progressive Resizing
            train_transform = A.Compose([
                        A.RandomResizedCrop(height=prog[ind], width=prog[ind], scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=1.0),
                        A.Flip(always_apply=False, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                        #A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            valid_transform = A.Compose([A.Resize(height=prog[ind], width=prog[ind], interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            pl_module.train_dataset = PANDADataset(pl_module.train_df, pl_module.hparams.data_dir, pl_module.hparams.image_format, transform=train_transform, tile=pl_module.hparams.tile, layer=pl_module.hparams.image_layer)
            pl_module.val_dataset = PANDADataset(pl_module.val_df, pl_module.hparams.data_dir, pl_module.hparams.image_format, transform=valid_transform, tile=pl_module.hparams.tile, layer=pl_module.hparams.image_layer)
            trainer.train_dataloader = DataLoader(pl_module.train_dataset, batch_size=batch[ind],
                                                shuffle=True, num_workers=4)
            trainer.val_dataloaders = [DataLoader(pl_module.train_dataset, batch_size=batch[ind],
                                                shuffle=True, num_workers=4)]
    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        # For Head First
        if trainer.current_epoch == pl_module.hparams.head_first & pl_module.hparams.head_first != 0:
            trainer.optimizers = [torch.optim.Adam(pl_module.model.parameters(), lr=pl_module.hparams.learning_rate * 0.5)]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizers[0], 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
            trainer.lr_schedulers = [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'reduce_on_plateau': True, 'monitor': 'avg_val_loss', }]
        # For log Learning Rate
        for param_group in trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        pl_module.logger.log_metrics({"learning_rate": lr})