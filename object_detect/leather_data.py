import os,cv2
import sys
import tarfile
import torch
import collections
import torch.utils.data as data
import shutil
import numpy as np
from object_detect.get_bboxes import convert_mask_to_bounding_box
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
        bboxes = []
        for i in range(np.shape(bounding_box)[0]):
            if bounding_box[i] == (0, 0, self.scale, self.scale):
                pass
            else:
                bboxes.append(bounding_box[i])

        if len(bboxes) == 0:
            bboxes.append((0, 0, self.scale, self.scale))
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            area = torch.zeros(1, dtype=torch.float32)
            labels = torch.zeros((len(bboxes),), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            labels = torch.ones((len(bboxes),), dtype=torch.int64)

        if self.transform is not None:
            img, target = self.transform(img, target)

        image_id = torch.tensor([index])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)

        targets = {}
        targets["boxes"] = boxes
        targets["labels"] = labels
        #targets["masks"] = tgt
        targets["image_id"] = image_id
        targets["area"] = area
        targets["iscrowd"] = iscrowd
        return img, targets, new_mask

    def __len__(self):
        return len(self.images)

