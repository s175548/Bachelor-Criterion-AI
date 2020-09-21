import os,cv2
import sys
import tarfile
import torch
import collections
import torch.utils.data as data
import shutil
import numpy as np
from data_import.masks_to_bounding_box import convert_mask_to_bounding_box
from PIL import Image




class LeatherData_BB(data.Dataset):

    def __init__(self,
                 path_mask,path_img,list_of_filenames,
                 transform=None):


        self.path_mask = path_mask
        self.path_img = path_img
        self.transform = transform


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
        new_mask = cv2.resize(mask, (512, 512))
        #Image._show(Image.fromarray(np.array(img)))
        #Image._show(Image.fromarray(mask))
        #Image._show(Image.fromarray(new_mask))
        #cv2.imshow('mask', mask)
        #Image._show(Image.fromarray(new_mask))
        #cv2.waitKey(0)
        bmask, bounding_box = convert_mask_to_bounding_box(new_mask)
        #bmask1, bounding_box1 = convert_mask_to_bounding_box(mask)
        #bmask1, bounding_box2 = convert_mask_to_bounding_box(np.resize(mask,(512,512,1) ))
        #cv2.imshow('bbox', bmask)
        #cv2.imshow('bbox',bmask1)
        #Image._show(Image.fromarray(bmask)) #<-
        #Image._show(Image.fromarray(bmask1))
        #cv2.waitKey(0)

        obj_ids = np.unique(new_mask)
        # first id is the background, so remove it
        #obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        image_id = torch.tensor([index])
        num_objs = len(obj_ids)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        bboxes = []
        for i in range(np.shape(bounding_box)[0]):
            if bounding_box[i] == (0, 0, 512, 512):
                pass
            else:
                bboxes.append(bounding_box[i])

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        if self.transform is not None:
            img, target = self.transform(img, target)
        tgt = torch.tensor(target, dtype=torch.uint8)
        targets = {}
        targets["boxes"] = boxes
        targets["labels"] = labels
        #targets["masks"] = torch.reshape(tgt,(1,512,512))
        targets["image_id"] = image_id
        targets["area"] = area
        targets["iscrowd"] = iscrowd
        return img, targets

    def __len__(self):
        return len(self.images)

