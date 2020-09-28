#
#     self.binary_class_dictionary = self.generate_binary_class_dictionary()
#
#
# def generate_binary_class_dictionary(self):
#         """     All categories found in metadata_csv are turned into dictionary, such that that can get a binary output (0: good, 1: defect) by parsing the category to the dict
#                 self.binary_class_dictionary[ self.metadata_csv[0,0] ] will return the binary value of the first datapoint.
#         """
#         binary_dict = {}
#         for ele in np.unique(self.metadata_csv[:, 0]):
#             if "good" in ele.lower():
#                 binary_dict[ele] = 0
#             else:
#                 binary_dict[ele] = 1
#         return binary_dict
#
#
#
#     def read_segmentation_file(self,filename):
#         """     Helper function, that simply opens segmentation file, draws a contour from this.
#                 Output: Segmentation retrieved from filename
#         """
#         fh = open(filename, "r")
#         try:
#             file_content = fh.read()
#             seg = json.loads(file_content)
#             segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
#             return segmentation
#         finally:
#             fh.close()
#
#             np.where(np.array([numb in dataloader.valid_annotations for numb in list(range(691))]) == False)[0] # Get index of invalid masks
#             np.where(np.array(dataloader.visibility_score) == 3)[0] #
#             np.sort(np.array(list(np.where(np.array(dataloader.visibility_score) == 3)[0]) + list(
#                 np.where(np.array(dataloader.visibility_score) == 2)[0])))


import sys
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')


from tqdm import tqdm
import random
import numpy as np
#from semantic_segmentation.DeepLabV3.dataset_class import LeatherData

from torch.utils import data
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import os
import PIL
import pickle
import matplotlib.pyplot as plt

model_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Downloads_hpc\Bachelor-Criterion-AImodel_pre_full.pt'
model = deeplabv3_resnet101()
model.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
model.aux_classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
device = torch.device('cpu')

if __name__ == "__main__":
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    del checkpoint
    print("Model restored")
    model()