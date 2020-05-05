#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import albumentations as A
import torch
from torch.utils.data import DataLoader

from dataset import PANDASegDataset
from loss import RMSELoss
import torch.nn as nn
import numpy as np

class PLImageSegmentationRegSystem(pl.LightningModule):
    
    def __init__(self, model, hparams):
    #def __init__(self, train_loader, val_loader, model):
        super(PLImageSegmentationRegSystem, self).__init__()
        #self.train_loader = train_loader
        #self.val_loader = val_loader
        self.hparams = hparams
        self.model = model
        #self.criteria = nn.MSELoss(reduction='sum')
        self.criteria = nn.NLLLoss()

    def forward(self, x):
        return self.model(x)
    
# For Training
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        log = {'avg_train_loss': avg_loss}
        return {'avg_train_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True, eps=1e-6)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'avg_val_loss'}]

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                    second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        self.logger.log_metrics({"learning_rate": lr})
    
# For Validation
    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log = {'avg_val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

# For Data
    def prepare_data(self):
        # Read DataFrame
        df = pd.read_csv(os.path.join(self.hparams.data_dir,'train.csv'))
        df = df[df['data_provider'] == 'radboud' ].reset_index(drop=True)
        #Delete rows (mask dont exist)
        for idx in range(len(df)):
            image_id = df.loc[idx, 'image_id']
            if not os.path.exists(os.path.join(os.path.join(self.hparams.data_dir, 'train_label_masks/'), image_id + '_mask.' + self.hparams.image_format)):
                df = df.drop(idx, axis=0)
        df = df.reset_index(drop=True)
        
        skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 2020)
        for fold, (train_index, val_index) in enumerate(skf.split(df.values, df['isup_grade'])):
            df.loc[val_index, 'fold'] = int(fold)
        df['fold'] = df['fold'].astype(int)
        #print(df)
        train_df = df[df['fold']!=self.hparams.fold]
        val_df = df[df['fold']==self.hparams.fold]
        #train_df, val_df = train_test_split(train_df, stratify=train_df['isup_grade'])

        train_transform = A.Compose([A.Resize(height=self.hparams.image_size, width=self.hparams.image_size, interpolation=1, always_apply=False, p=1.0),
                     A.Flip(always_apply=False, p=0.5),
                     A.RandomResizedCrop(height=self.hparams.image_size, width=self.hparams.image_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=1.0),
                     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                     A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                     #A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                      ])

        valid_transform = A.Compose([A.Resize(height=self.hparams.image_size, width=self.hparams.image_size, interpolation=1, always_apply=False, p=1.0),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                      ])

        self.train_dataset = PANDASegDataset(train_df, self.hparams.data_dir, self.hparams.image_format, transform=train_transform)
        self.val_dataset = PANDASegDataset(val_df, self.hparams.data_dir, self.hparams.image_format, transform=valid_transform)
        
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=4)

    #def test_dataloader(self):
    #    # OPTIONAL
    #    pass

def preds_rounder(test_preds):
    coef = [0.5, 1.5, 2.5, 3.5, 4.5]

    for i, pred in enumerate(test_preds):
        if pred < coef[0]:
            test_preds[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            test_preds[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            test_preds[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            test_preds[i] = 3
        elif pred >= coef[3] and pred < coef[4]:
            test_preds[i] = 4
        else:
            test_preds[i] = 5
    return test_preds



