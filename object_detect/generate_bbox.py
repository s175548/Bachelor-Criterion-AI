import os,cv2
import sys
import tarfile
import torch
import collections
import torch.utils.data as data
import shutil
import numpy as np
from Bachelor-Criterion-AI.object_detect.get_bboxes import convert_mask_to_bounding_box
from PIL import Image




class LeatherData_BB(data.Dataset):

    def __init__(self,
                 path_mask,path_img,list_of_filenames,scale,
                 transform=None):


        self.path_mask = path_mask
        self.path_img = path_img
        self.transform = transform
        self.scale = scale


        file_names_mask=os.listdir(self.path_mask)
        file_names_img=os.listdir(self.path_img)
        file_names=list_of_filenames

        self.images = [os.path.join(self.path_img, x+ '.png') for x in file_names]
        self.masks = [os.path.join(self.path_mask, x+ '_mask.png') for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('P')

        mask = np.array(target)
        new_mask = cv2.resize(mask, (self.scale, self.scale))
        bmask, bounding_box = convert_mask_to_bounding_box(new_mask)

        return new_mask, bmask, bounding_box

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    print("HPC ")
    txt = "HOC"
    np.save(txt)