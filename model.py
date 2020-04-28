#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torchvision.models as models
import pretrainedmodels
import torch.nn as nn

def get_model_from_name(model_name=None, image_size=None, num_classes=None, pretrained=True):
    
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