# %% [code]
from __future__ import print_function, division
import albumentations as A
from PIL import ImageEnhance

from glob import glob
import cv2
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint, uniform
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import rotate
from torchvision.transforms import v2
# Ignore warnings
import warnings
import pdb
import imgaug.augmenters as iaa

warnings.filterwarnings("ignore")
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def create_image_mask_paths(path, image_ext='png', mask_ext='png'):
    image_paths = glob(os.path.join(path, "images", f"*.{image_ext}"))
    mask_paths = glob(os.path.join(path, "masks", f"*.{mask_ext}"))

    image_ids = [image_path.split("_")[1] for image_path in image_paths]
    mask_ids = [mask_path.split("_")[1] for mask_path in mask_paths]

    img_dict = {'train': [], 'test': [], 'val': []}
    mask_dict = {'train': [], 'test': [], 'val': []}

    val_ids = [8, 9]
    test_ids = [10, 11]

    for image_path, image_id, mask_path, mask_id in zip(image_paths, image_ids, mask_paths, mask_ids):
        image_id = int(image_id)
        mask_id = int(mask_id)

        if image_id in test_ids and mask_id in test_ids:
            img_dict['test'].append(image_path)
            mask_dict['test'].append(mask_path)

        elif image_id in val_ids and mask_id in val_ids:
            img_dict['val'].append(image_path)
            mask_dict['val'].append(mask_path)
        else:
            img_dict['train'].append(image_path)
            mask_dict['train'].append(mask_path)

    return img_dict, mask_dict

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    image_dict, mask_dict = create_image_mask_paths(root)
    items = []
    if mode == 'train':
        images = image_dict['train']
        labels = mask_dict['train']

        images.sort()
        labels.sort()
        
        for it_im, it_gt in zip(images, labels):
            item = it_im, it_gt
            items.append(item)
    elif mode == 'val':
        images = image_dict['val']
        labels = mask_dict['val']

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = it_im, it_gt
            items.append(item)
    else:
        images = image_dict['test']
        labels = mask_dict['test']

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = it_im, it_gt
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """CHAOS dataset."""

    def __init__(self, mode,noise_typ, root_dir = '/kaggle/input/cis-data/data', augment=False, noise=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.Noise = noise
        self.noise_typ = noise_typ
        self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.imgs)
    
    def noisy(self,image,noise_typ):
       if noise_typ == "gauss":
          row,col,ch= image.shape
          mean = 0
          var = 0.02
          sigma = var**0.5
          gauss = np.random.normal(mean,sigma,(row,col,ch))
          gauss = gauss.reshape(row,col,ch)
          noisy = image + gauss
          return noisy
       elif noise_typ =="speckle":
          row,col,ch = image.shape
          gauss = np.random.randn(row,col,ch)
          gauss = gauss.reshape(row,col,ch)        
          noisy = image + image * gauss
       return noisy

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        #---------------------------------------------------------------------
        img = cv2.imread(img_path) 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        T = A.Resize(height=256, width=256)
        resized_image = T(image=img, mask = mask)
        img = resized_image['image']
        mask = resized_image['mask']
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        #---------------------------------------------------------------------
        print(img.shape)
        print(mask.shape)
        if self.augmentation:
            transformed = TRAIN_VAL_TRANSFORM(image=img.astype(np.float32), mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        #---------------------------------------------------------------------
        if self.Noise:
            if uniform(0,1) > .72:
                img = self.noisy(img, self.noise_typ)
            
        #---------------------------------------------------------------------
        img = torch.from_numpy(img.reshape(3,256,256).astype(np.float32))
        mask = torch.from_numpy((mask.reshape(1,256,256)/255).astype(np.float32))
        #---------------------------------------------------------------------
      
        return [img, mask]
    
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
# Define the transformations
TRAIN_VAL_TRANSFORM = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),  # Randomly change image gamma
    A.Blur(p=0.1),  # Apply random blur to the image
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0),  # Apply elastic transformation
    A.GridDistortion(p=0.5),  # Apply grid distortion
    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=.5),  # Apply optical distortion
])