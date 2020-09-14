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
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 path_mask,path_img,
                 transform=None):


        self.path_mask = path_mask
        self.path_img = path_img
        self.transform = transform


        if not os.path.isdir(path):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')



        file_names=os.listdir()

        self.images = [os.path.join(self.path_img, x + ".jpg") for x[:-4] in file_names]
        self.masks = [os.path.join(self.path_mask, x + ".png") for x[:-4] in file_names]
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

