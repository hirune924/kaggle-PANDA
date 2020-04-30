#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict


def load_pytorch_model(ckpt_name, model):
    state_dict = torch.load(ckpt_name)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('model.'):
            name = name.replace('model.', '') # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model