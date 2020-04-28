#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torchvision.models as models
import pretrainedmodels
import segmentation_models_pytorch as smp
import torch.nn as nn

def get_cls_model_from_name(model_name=None, image_size=None, num_classes=None, pretrained=True):
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'se_resnet50':
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
    else:
        print('{} is not implimented'.format(model_name))
        model = None

    return model


def get_seg_model_from_name(model_name=None, in_channels=None, image_size=None, num_classes=None, pretrained=True):
    
    if model_name == 'resnet18_unet':
        model = smp.Unet(encoder_name='resnet18', in_channels=in_channels, classes=num_classes, activation=None, encoder_weights='imagenet')

    elif model_name == 'resnet34_unet':
        model = smp.Unet('resnet34', encoder_weights='imagenet')
    else:
        print('{} is not implimented'.format(model_name))
        model = None

    return model