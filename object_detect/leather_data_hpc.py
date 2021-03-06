import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
from data_import.masks_to_bounding_box import convert_mask_to_bounding_box
from PIL import Image
from object_detect.get_bboxes import new_convert, convert_mask_to_bbox, get_multi_bboxes
import torch





class LeatherDataZ(data.Dataset):

    def __init__(self,
                 path_mask,path_img,list_of_filenames,bbox=False,multi=False,
                 transform=None,color_dict=None,target_dict=None,unlabelled =False):


        self.path_mask = path_mask
        self.path_img = path_img
        self.transform = transform
        self.color_dict=color_dict
        self.target_dict=target_dict
        self.bbox = bbox
        self.multi = multi
        self.unlabelled = unlabelled


        file_names_mask=os.listdir(self.path_mask)
        file_names_img=os.listdir(self.path_img)
        file_names=list_of_filenames

        self.images = [os.path.join(self.path_img, x + '_img.png') for x in file_names]
        self.masks = [os.path.join(self.path_mask, x + '_mask.png') for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if self.unlabelled:
            img = Image.open(self.images[index]).convert('RGB')
            w, h = img.size
            target = Image.fromarray(np.empty( (img.size)) )#target is not needed
            img, target = self.transform(img,target)
            return img,target

        img = Image.open(self.images[index]).convert('RGB')
        target = np.array(Image.open(self.masks[index]))
        mask_for_bbox = Image.open(self.masks[index]).convert('L')
        img_for_bbox = Image.open(self.images[index]).convert('RGB')
        img_index = index

        for key, value in self.target_dict.items():
            value = self.color_dict[key]
            index = (target[:, :, 0] == value[0]) & (target[:, :, 1] == value[1]) & (target[:, :, 2] == value[2])
            target[index, :] = self.target_dict[key]

        if self.bbox == True:
            if self.transform is not None:
                img, tgt = self.transform(img_for_bbox, mask_for_bbox)
            if self.transform is not None:
                target2 = Image.fromarray(target)
                img2, tgt2 = self.transform(img_for_bbox, target2)
            if self.transform is not None:
                target3 = Image.fromarray(target).convert('L')
                img3, tgt3 = self.transform(img_for_bbox, target3)
            mask = tgt.numpy()
            mask2 = tgt2[:,:,0].numpy()
            mask3 = tgt3.numpy()
            if self.multi == True:
                bmask, bounding_box, bbox_labels, _ = get_multi_bboxes(mask3)
            else:
                bmask, bounding_box = new_convert(mask)
            bboxes = bounding_box
            #for i in range(np.shape(bounding_box)[0]):
            #   bboxes.append(bounding_box[i])
            if len(bboxes) == 0:
                #bboxes.append((0, 0, 255, 255))
                #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                empty_box = [np.nan, np.nan, np.nan, np.nan]
                empty_box2 = torch.tensor(list(empty_box), dtype=torch.float32)
                area = []
                area = torch.tensor(area)
                labels = []
                labels = torch.tensor(labels)
                # suppose all instances are not crowdpr
                iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)
                targets = {}
                image_id = torch.tensor([img_index])
                targets["boxes"] = torch.tensor(bboxes,dtype=torch.float32)
                targets["labels"] = torch.zeros(1, dtype=torch.int64)
                targets["image_id"] = image_id
                return img, targets, mask
            else:
                boxes = torch.as_tensor(bboxes, dtype=torch.float32)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                if self.multi:
                    labels = torch.tensor(bbox_labels, dtype=torch.int64)
                else:
                    labels = torch.ones((len(bboxes),), dtype=torch.int64)
            image_id = torch.tensor([img_index])
            # suppose all instances are not crowdpr
            iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)
            targets = {}
            targets["boxes"] = boxes
            targets["labels"] = labels
            targets["image_id"] = image_id
            targets["area"] = area
            targets["iscrowd"] = iscrowd
            return img, targets, mask


        if self.transform is not None:
            target = Image.fromarray(target)
            img, target = self.transform(img, target)


        return img, target[:,:,0]

    def __len__(self):
        return len(self.images)

