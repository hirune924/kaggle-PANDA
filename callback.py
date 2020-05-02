#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl

class MyCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        print(trainer.current_epoch)
        if trainer.current_epoch == 1:
            trainer.optimizers = [torch.optim.Adam(pl_module.model.parameters(), lr=pl_module.hparams.learning_rate)]
        for param_group in trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        pl_module.logger.log_metrics({"learning_rate": lr})