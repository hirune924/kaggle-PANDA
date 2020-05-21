import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
from tqdm import tqdm
import skimage.io
from skimage.transform import resize, rescale
from argparse import ArgumentParser
import shutil

import numpy as np

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

def load_img(img_name, layer):
    image = skimage.io.MultiImage(img_name)[layer]
    
    image = tile(image, sz=2048, N=16)
    #image = np.random.shuffle(image))
    image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
                     cv2.vconcat([image[4], image[5], image[6], image[7]]), 
                     cv2.vconcat([image[8], image[9], image[10], image[11]]), 
                     cv2.vconcat([image[12], image[13], image[14], image[15]])])
    return image

def main(args):
    
    train_labels = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))

    os.makedirs(os.path.join(args.save_dir, 'train_images'), exist_ok=True)
    shutil.copyfile(os.path.join(args.data_dir, 'sample_submission.csv'),os.path.join(args.save_dir, 'sample_submission.csv'))
    shutil.copyfile(os.path.join(args.data_dir, 'train.csv'),os.path.join(args.save_dir, 'train.csv'))
    shutil.copyfile(os.path.join(args.data_dir, 'test.csv'),os.path.join(args.save_dir, 'test.csv'))
    
    for img_id in tqdm(train_labels.image_id):
        load_path = os.path.join(args.data_dir, 'train_images/' + img_id + '.tiff')
        save_path = os.path.join(args.save_dir, 'train_images/' + img_id + '.png')

        #biopsy = skimage.io.MultiImage(load_path)
        biopsy_tile = load_img(load_path, 0)
        img = cv2.resize(biopsy_tile, (args.size, args.size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)

    os.makedirs(os.path.join(args.save_dir, 'train_label_masks'), exist_ok=True)
    mask_files = os.listdir(os.path.join(args.data_dir, 'train_label_masks'))
    
    for mask_file in tqdm(mask_files):
        load_path = os.path.join(args.data_dir, 'train_label_masks/' + mask_file)
        save_path = os.path.join(args.save_dir, 'train_label_masks/' + mask_file.replace('.tiff', '.png'))

        #mask = skimage.io.MultiImage(load_path)[0]
        mask_tile = load_img(load_path, 0)
        img = cv2.resize(mask_tile, (args.size, args.size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)
        
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--size', help='image size',
                        type=int, required=False, default=256)
    parser.add_argument('-sd', '--save_dir', help='path to log',
                        type=str, required=True)
    parser.add_argument('-dd', '--data_dir', help='path to data dir',
                        type=str, required=True)
    
    #args = parser.parse_args(['-dd', '../input/prostate-cancer-grade-assessment/', '-sd','../working'])
    args = parser.parse_args()

    main(args)
