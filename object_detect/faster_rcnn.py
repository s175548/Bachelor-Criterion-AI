from tqdm import tqdm
import torchvision
#import utils
import random
import argparse
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData

from torch.utils import data
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
#from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.vgg import vgg16
from torchvision.models.detection.faster_rcnn import FasterRCNN
from semantic_segmentation.DeepLabV3.Training import get_argparser, validate

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
backbone = vgg16(pretrained=True).features
model=FasterRCNN(backbone,
                 pretrained=True,
                 num_classes=2,
                 rpn_anchor_generator=anchor_generator,
                 box_roi_pool=roi_pooler)
for param in model.parameters():
    param.requires_grad = False
# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNN(in_features, num_classes=2)
total_itrs=10
lr=0.01
lr_policy='step'
step_size=10000
batch_size=16
val_batch_size=4
loss_type="cross_entropy"
weight_decay=1e-4
random_seed=1
print_interval=10
val_interval=100
vis_num_samples=2
enable_vis=True

