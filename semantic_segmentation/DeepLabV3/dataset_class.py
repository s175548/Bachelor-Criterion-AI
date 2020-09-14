import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity



class LeatherData(data.Dataset):

    def __init__(self,
                 path_mask,path_img,
                 transform=None):


        self.path_mask = path_mask
        self.path_img = path_img
        self.transform = transform


        file_names_mask=os.listdir(self.path_mask)
        file_names_img=os.listdir(self.path_img)

        self.images = [os.path.join(self.path_img, x) for x in file_names_img]
        self.masks = [os.path.join(self.path_mask, x) for x in file_names_mask]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

