import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io
import gzip
import wget
import h5py
import pickle
import urllib
import os
import skimage
import skimage.transform
from skimage.io import imread
import matplotlib.image as mpimg
from glob import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize

def LoadDataset(name, root, batch_size, split, shuffle=True, style=None, attr=None):
    if name == 'face':
        assert style != None  # style -- which group
        if split == 'train':
            return LoadFace(root, style=style, split='train', batch_size=batch_size, shuffle=shuffle)
        elif split == 'test':
            return LoadFace(root, style=style, split='test', batch_size=batch_size, shuffle=False)

def LoadFace(data_root, batch_size=32, split='train', style='photo', attr=None,
             shuffle=True, load_first_n=None):  # style -- a1,a2,a3
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    key = '/'.join([split, style])
    UTKF_dataset = Face(data_root, key, split,load_first_n)
    return DataLoader(UTKF_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

class Face(Dataset):
    def __init__(self, root, key,split, load_first_n=None):
        resize_size=64
        crop_size=64
        transforms = [Resize((resize_size, resize_size), Image.BICUBIC)]
        if split == 'train':
            transforms.append(RandomCrop(crop_size))
        else:
            transforms.append(CenterCrop(crop_size))
        # if not opts.no_flip:
        #     transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        data_dir = glob(os.path.join(root, key,'*.jpg'))
        # # data=
        # with h5py.File(root,'r') as f:
        #     data = f[key][()]
        #     if load_first_n:
        #         data = data[:load_first_n]
        self.imgs = data_dir

    def __getitem__(self, index):
        return self.load_img(self.imgs[index])

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imgs)