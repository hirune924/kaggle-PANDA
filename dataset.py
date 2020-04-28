#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import skimage.io
import os
import cv2

class PANDADataset(Dataset):
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
        data_provider = self.data.loc[idx, 'data_provider']
        gleason_score = self.data.loc[idx, 'gleason_score']
        isup_grade = label = self.data.loc[idx, 'isup_grade']
        
        if self.image_format == 'tiff':
            image = skimage.io.MultiImage(img_name)[-1]
        elif self.image_format == 'png':
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)
            image = torch.from_numpy(image['image'].transpose(2, 0, 1))
        return image, isup_grade