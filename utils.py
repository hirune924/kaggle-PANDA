#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict

import random
import cv2
import numpy as np

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

def crop_tile(image_org):
    image = image_org
    img_y,img_x=image.shape[:2]
    y_step=244#int(img_y/3) #高さ方向のグリッド間隔(単位はピクセル)
    x_step=244#int(img_x/3) #幅方向のグリッド間隔(単位はピクセル)
    #横線を引く：y_stepからimg_yの手前までy_stepおきに白い(BGRすべて255)横線を引く
    image[y_step:img_y:y_step, :, :] = 255
    #縦線を引く：x_stepからimg_xの手前までx_stepおきに白い(BGRすべて255)縦線を引く
    image[:, x_step:img_x:x_step, :] = 255
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours,hierarchy = cv2.findContours(255-gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objs = []
    h_all = 0
    w_max = 0
    for cnt in contours:
        if cv2.contourArea(cnt)<50:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        im = image[y:y+h, x:x+w]
        if w>h:
            h_all += w
            w_max = max(w_max, h)
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        else:
            h_all += h
            w_max = max(w_max, w)
        objs.append(im)
    if w_max == 0:
        return image_org
    random.shuffle(objs)  
    bg = np.zeros((h_all, w_max, 3), np.uint8)+255
    h_loc = 0
    for obj in objs:
        h,w,c = obj.shape
        bg[h_loc:h_loc+h, 0:w,:]=obj
        h_loc += h
    
    nsp = int(np.floor(np.sqrt(h_all/w_max)))
    hsp = int(h_all/nsp)+10
    result = np.zeros((hsp, int(w_max*nsp), 3), np.uint8)+255
    for idx in range(nsp):
        if hsp*(idx+1)<h_all:
            result[0:hsp, w_max*idx:w_max*(idx+1)] = bg[hsp*idx:hsp*(idx+1),0:w_max]
        else:
            result[0:h_all-hsp*idx, w_max*idx:w_max*(idx+1)] = bg[hsp*idx:h_all,0:w_max]
    return result


def tile(img, sz=128, N=16):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img