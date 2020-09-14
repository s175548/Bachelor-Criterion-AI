import torch
import utils
import random
import argparse
import matplotlib.patches as patches
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData

from torch.utils import data
from metrics import StreamSegMetrics

from torch

import torch
import torch.nn as nn
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
