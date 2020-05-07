#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import skimage.io
import os
import cv2

from utils import crop_tile

class PANDADataset(Dataset):
    """PANDA Dataset."""
    
    def __init__(self, dataframe, data_dir, image_format, transform=None, tile=False, layer=-1):
        """
        Args:
            data_path (string): data path(glob_pattern) for dataset images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataframe.reset_index(drop=True) #pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
        self.transform = transform
        self.data_dir = data_dir
        self.image_format = image_format
        self.tile = tile
        self.layer = layer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(os.path.join(self.data_dir, 'train_images/'), self.data.loc[idx, 'image_id'] + '.' +self.image_format)
        data_provider = self.data.loc[idx, 'data_provider']
        gleason_score = self.data.loc[idx, 'gleason_score']
        isup_grade = label = self.data.loc[idx, 'isup_grade']
        
        if self.image_format == 'tiff':
            image = skimage.io.MultiImage(img_name)[self.layer]
        elif self.image_format == 'png':
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.tile:
            try:
                image = crop_tile(image)
            except ZeroDivisionError:
                print(img_name)
        if self.transform:
            image = self.transform(image=image)
            image = torch.from_numpy(image['image'].transpose(2, 0, 1))
        return image, isup_grade



class PANDASegDataset(Dataset):
    """PANDA Dataset."""
    
    def __init__(self, dataframe, data_dir, image_format, transform=None):
        """
        Args:
            data_path (string): data path(glob_pattern) for dataset images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataframe.reset_index(drop=True) #pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
        self.transform = transform
        self.data_dir = data_dir
        self.image_format = image_format
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(os.path.join(self.data_dir, 'train_images/'), self.data.loc[idx, 'image_id'] + '.' +self.image_format)
        mask_name = os.path.join(os.path.join(self.data_dir, 'train_label_masks/'), self.data.loc[idx, 'image_id'] + '_mask.' +self.image_format)

        data_provider = self.data.loc[idx, 'data_provider']
        gleason_score = self.data.loc[idx, 'gleason_score']
        isup_grade = label = self.data.loc[idx, 'isup_grade']
        
        if self.image_format == 'tiff':
            image = skimage.io.MultiImage(img_name)[-1]
            mask = skimage.io.MultiImage(mask_name)[-1]
        elif self.image_format == 'png':
            image = cv2.imread(img_name)
            mask = cv2.imread(mask_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:,:,0]
        
        if self.transform:
            #print(torch.max(torch.from_numpy(mask)))
            trns = self.transform(image=image, mask=mask)
            image = trns['image']
            mask = trns['mask']
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask).long()
        #if data_provider=='karolinska':
        #    image = torch.cat([image, torch.ones_like(mask), torch.zeros_like(mask)], dim=0)
        #    mask = torch.cat([mask, torch.zeros_like(mask)], dim=0)
        #elif data_provider=='radboud':
        #    image = torch.cat([image, torch.zeros_like(mask), torch.ones_like(mask)], dim=0)
        #    mask = torch.cat([torch.zeros_like(mask), mask], dim=0)
        return image, mask


class PANDASegClsDataset(Dataset):
    """PANDA Dataset."""
    
    def __init__(self, dataframe, data_dir, image_format, transform=None):
        """
        Args:
            data_path (string): data path(glob_pattern) for dataset images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataframe.reset_index(drop=True) #pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
        self.transform = transform
        self.data_dir = data_dir
        self.image_format = image_format
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(os.path.join(self.data_dir, 'train_images/'), self.data.loc[idx, 'image_id'] + '.' +self.image_format)
        mask_name = os.path.join(os.path.join(self.data_dir, 'train_label_masks/'), self.data.loc[idx, 'image_id'] + '_mask.' +self.image_format)

        data_provider = self.data.loc[idx, 'data_provider']
        gleason_score = self.data.loc[idx, 'gleason_score']
        isup_grade = label = self.data.loc[idx, 'isup_grade']
        
        if self.image_format == 'tiff':
            image = skimage.io.MultiImage(img_name)[-1]
        elif self.image_format == 'png':
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            trns = self.transform(image=image)
            image = trns['image']
            image = torch.from_numpy(image.transpose(2, 0, 1))
        ch, height, width = image.shape
        if data_provider=='karolinska':
            image = torch.cat([image, torch.ones([1, height, width]), torch.zeros([1, height, width])], dim=0)
        elif data_provider=='radboud':
            image = torch.cat([image, torch.zeros([1, height, width]), torch.ones([1, height, width])], dim=0)
        return image, isup_grade
