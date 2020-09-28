import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
from data_import.masks_to_bounding_box import convert_mask_to_bounding_box
from PIL import Image




class LeatherData(data.Dataset):

    def __init__(self,
                 path_mask,path_img,list_of_filenames,
                 transform=None):


        self.path_mask = path_mask
        self.path_img = path_img
        self.transform = transform


        file_names_mask=os.listdir(self.path_mask)
        file_names_img=os.listdir(self.path_img)
        file_names=list_of_filenames

        self.images = [os.path.join(self.path_img, x+ '.jpg') for x in file_names]
        self.masks = [os.path.join(self.path_mask, x+ '_mask.png') for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index, bounding_box=False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if bounding_box == True:
            _, bounding_box = convert_mask_to_bounding_box(mask)
            mask = np.array(target)
            objs = np.unique(mask)
            masks = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)
            boxes = torch.as_tensor(bounding_box, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            if self.transform is not None:
                img, target = self.transform(img, target)
            targets = {}
            targets["boxes"] = boxes
            targets["labels"] = labels
            targets["masks"] = masks
            return img, targets

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

