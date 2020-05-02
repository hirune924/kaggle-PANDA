#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl

class MyCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        
        if trainer.current_epoch == pl_module.hparams.head_first & pl_module.hparams.head_first != 0:
            trainer.optimizers = [torch.optim.Adam(pl_module.model.parameters(), lr=pl_module.hparams.learning_rate * 0.5)]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizers[0], 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
            trainer.lr_schedulers = [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'reduce_on_plateau': True, 'monitor': 'avg_val_loss', }]
        for param_group in trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        pl_module.logger.log_metrics({"learning_rate": lr})