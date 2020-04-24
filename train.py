#!pip install pytorch-lightning
#!pip install neptune-client

from argparse import ArgumentParser
# for dataset and dataloader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
import glob
import os
import numpy as np

import pandas as pd
import skimage.io
import albumentations as A

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning.logging import CometLogger
from pytorch_lightning import loggers

import torchvision.models as models
import torch.nn as nn


# for visualization
#import matplotlib.pyplot as plt
#%matplotlib inline

class PANDADataset(Dataset):
    """PANDA Dataset."""
    
    def __init__(self, dataframe, data_dir, transform=None):
        """
        Args:
            data_path (string): data path(glob_pattern) for dataset images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataframe.reset_index(drop=True) #pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(os.path.join(self.data_dir, 'train_images/'), self.data.loc[idx, 'image_id'] + '.tiff')
        data_provider = self.data.loc[idx, 'data_provider']
        gleason_score = self.data.loc[idx, 'gleason_score']
        isup_grade = label = self.data.loc[idx, 'isup_grade']
        
        image = skimage.io.MultiImage(img_name)[-1]
        #print(img_name)
        
        if self.transform:
            image = self.transform(image=image)
            image = torch.from_numpy(image['image'].transpose(2, 0, 1))
        return image, isup_grade



class PLBasicImageClassificationSystem(pl.LightningModule):
    
    def __init__(self, model, hparams):
    #def __init__(self, train_loader, val_loader, model):
        super(PLBasicImageClassificationSystem, self).__init__()
        #self.train_loader = train_loader
        #self.val_loader = val_loader
        self.hparams = hparams
        self.model = model
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
# For Training
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        return {'loss': loss}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
    
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
        return {'avg_val_loss': avg_loss}

# For Data
    def prepare_data(self):
        train_df = pd.read_csv(os.path.join(self.hparams.data_dir,'train.csv'))
        train_df, val_df = train_test_split(train_df, stratify=train_df['isup_grade'])

        transform = A.Compose([A.Resize(height=self.hparams.image_size, width=self.hparams.image_size, interpolation=1, always_apply=False, p=1),
                     A.Flip(always_apply=False, p=0.5),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                      ])
        
        self.train_dataset = PANDADataset(train_df, self.hparams.data_dir, transform=transform)
        self.val_dataset = PANDADataset(val_df, self.hparams.data_dir, transform=transform)
        
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=4)

    #def test_dataloader(self):
    #    # OPTIONAL
    #    pass


def get_model_from_name(model_name=None, image_size=None, num_classes=None, pretrained=True):
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        print('{} is not implimented'.format(model_name))
        model = None

    return model



def main(hparams):
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2I2ZWM0NmQtNjg0NS00ZjM5LTkzNTItN2I4Nzc0YTUzMmM0In0=",
        project_name="hirune924/kaggle-PANDA",
        close_after_fit=False,
        upload_source_files=['*.py','*.ipynb'],
        params=vars(hparams)
        #experiment_name="default",  # Optional,
        #tags=["pytorch-lightning", "mlp"]  # Optional,
    )
    comet_logger = CometLogger(
        api_key="QCxbRVX2qhQj1t0ajIZl2nk2c",
        workspace='hirune924',  # Optional
        save_dir='.',  # Optional
        project_name="kaggle-panda",  # Optional
        #rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
        #experiment_name='default'  # Optional
    )
    tb_logger = loggers.TensorBoardLogger(save_dir=hparams.log_dir, name='default', version=None)

    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.log_dir,
        save_top_k=10,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        save_weights_only = True,
        period = 1
    )

    # default used by the Trainer
    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        patience=5,
        min_delta = 0.0,
        strict=False,
        verbose=False,
        mode='min'
    )

    model = get_model_from_name(model_name='resnet18', num_classes=6, pretrained=True)
    pl_model = PLBasicImageClassificationSystem(model, hparams)

    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs,min_epochs=hparams.min_epochs,
                    max_steps=None,min_steps=None,
                    checkpoint_callback=checkpoint_callback,
                    early_stop_callback=early_stop_callback,
                    logger=tb_logger,#[tb_logger, neptune_logger],
                    accumulate_grad_batches=1,
                    precision=hparams.precision,
                    amp_level='O1',
                    auto_lr_find=True,
                    benchmark=True,
                    check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                    distributed_backend=hparams.distributed_backend,
                    num_nodes=1,
                    fast_dev_run=False,
                    gradient_clip_val=0.0,
                    log_gpu_memory=None,
                    log_save_interval=100,
                    num_sanity_val_steps=5,
                    overfit_pct=0.0)

    # fit model !
    trainer.fit(pl_model)

    neptune_logger.experiment.log_artifact(hparams.log_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-ng', '--gpus', help='num GPU all=-1',
                        type=int, required=False, default=-1)
    parser.add_argument('-pr', '--precision', help='precision',
                        type=int, required=False, default=32)
    parser.add_argument('-emax', '--max_epochs', help='max_epochs',
                        type=int, required=False, default=100)
    parser.add_argument('-emin', '--min_epochs', help='min_epochs',
                        type=int, required=False, default=10)
    parser.add_argument('-eval', '--check_val_every_n_epoch', help='check_val_every_n_epoch',
                        type=int, required=False, default=1)
    parser.add_argument('-bs', '--batch_size', help='batch_size',
                        type=int, required=False, default=32)
    parser.add_argument('-is', '--image_size', help='image_size',
                        type=int, required=False, default=256)
    parser.add_argument('-lr', '--learning_rate', help='learning_rate',
                        type=float, required=False, default=0.001)
    parser.add_argument('-db', '--distributed_backend', help='distributed_backend',
                        type=str, required=False, default='dp')
    parser.add_argument('-ld', '--log_dir', help='path to log',
                        type=str, required=True)
    parser.add_argument('-dd', '--data_dir', help='path to data dir',
                        type=str, required=True)
    
    #args = parser.parse_args(['-ld', '../working/', '-dd','../input/prostate-cancer-grade-assessment/'])
    args = parser.parse_args()

    main(args)