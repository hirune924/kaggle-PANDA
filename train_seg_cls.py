#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!pip install pytorch-lightning
#!pip install neptune-client

from argparse import ArgumentParser
# for dataset and dataloader
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning import loggers
import glob
#from pytorch_lightning.logging import CometLogger


from utils import load_pytorch_model
from model import get_seg_model_from_name, get_cls_model_from_name
from systems_seg_cls import PLImageSegmentationClassificationSystem


def main(hparams):
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2I2ZWM0NmQtNjg0NS00ZjM5LTkzNTItN2I4Nzc0YTUzMmM0In0=",
        project_name="hirune924/kaggle-PANDA",
        close_after_fit=False,
        upload_source_files=['*.py','*.ipynb'],
        params=vars(hparams),
        experiment_name=hparams.experiment_name,  # Optional,
        #tags=["pytorch-lightning", "mlp"]  # Optional,
    )
    '''
    comet_logger = CometLogger(
        api_key="QCxbRVX2qhQj1t0ajIZl2nk2c",
        workspace='hirune924',  # Optional
        save_dir='.',  # Optional
        project_name="kaggle-panda",  # Optional
        #rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
        #experiment_name='default'  # Optional
    )'''
    tb_logger = loggers.TensorBoardLogger(save_dir=hparams.log_dir, name='default', version=None)
 
    logger_list = [tb_logger, neptune_logger] if hparams.distributed_backend!='ddp' else tb_logger

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(hparams.log_dir, '{epoch}-{avg_val_loss}-{val_qwk}'),
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
        patience=20,
        min_delta = 0.0,
        strict=True,
        verbose=True,
        mode='min'
    )

    seg_model = get_seg_model_from_name(model_name=hparams.seg_model_name, in_channels=5, num_classes=2, pretrained=True)
    seg_ckpt_pth = glob.glob(os.path.join(hparams.seg_ckpt_dir,'fold'+str(hparams.fold)+'*.ckpt'))
    seg_model = load_pytorch_model(seg_ckpt_pth[0], seg_model)
    if hparams.marge_type == 'cat':
        in_channels = 7     
    elif hparams.marge_type == 'add':
        in_channels =  3
    cls_model = get_cls_model_from_name(model_name=hparams.cls_model_name, in_channels=in_channels, num_classes=1, pretrained=True)
    pl_model = PLImageSegmentationClassificationSystem(seg_model, cls_model, hparams)

###
    if hparams.auto_lr_find:
        trainer = Trainer()
        lr_finder = trainer.lr_find(pl_model)
        print(lr_finder.results)
        print(lr_finder.suggestion())
        pl_model.learning_rate = lr_finder.suggestion()
###

    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs,min_epochs=hparams.min_epochs,
                    max_steps=None,min_steps=None,
                    checkpoint_callback=checkpoint_callback,
                    early_stop_callback=early_stop_callback,
                    logger=logger_list,
                    accumulate_grad_batches=1,
                    precision=hparams.precision,
                    amp_level='O1',
                    auto_lr_find=False,
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

    #neptune_logger.experiment.log_artifact(hparams.log_dir)


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
    parser.add_argument('-nf', '--num_fold', help='fold num',
                        type=int, required=False, default=5)
    parser.add_argument('-f', '--fold', help='target fold',
                        type=int, required=False, default=0)
                        
    parser.add_argument('-alf', '--auto_lr_find', help='auto lr find.', 
                        action='store_true')

    parser.add_argument('-lr', '--learning_rate', help='learning_rate',
                        type=float, required=False, default=1e-4)

    parser.add_argument('-db', '--distributed_backend', help='distributed_backend',
                        type=str, required=False, default='dp')
    parser.add_argument('-if', '--image_format', help='image_format',
                        type=str, required=False, default='tiff')
    parser.add_argument('-smn', '--seg_model_name', help='seg_model_name',
                        type=str, required=False, default='resnet18_unet')
    parser.add_argument('-scd', '--seg_ckpt_dir', help='seg_ckpt_dir',
                        type=str, required=False, default=None)
    parser.add_argument('-cmn', '--cls_model_name', help='cls_model_name',
                        type=str, required=False, default='resnet18')
    parser.add_argument('-mt', '--marge_type', help='marge_type',
                        type=str, required=False, default='cat')
    parser.add_argument('-en', '--experiment_name', help='experiment_name',
                        type=str, required=False, default='default')
    parser.add_argument('-ld', '--log_dir', help='path to log',
                        type=str, required=True)
    parser.add_argument('-dd', '--data_dir', help='path to data dir',
                        type=str, required=True)
    
    #args = parser.parse_args(['-ld', '../working/', '-dd','../input/prostate-cancer-grade-assessment/'])
    args = parser.parse_args()

    main(args)
