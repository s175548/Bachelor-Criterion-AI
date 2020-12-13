from semantic_segmentation.DeepLabV3.network.modeling import _segm_mobilenet
from torchvision.models.segmentation import deeplabv3_resnet101
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model=deeplabv3_resnet101(pretrained=True, progress=True,num_classes=21, aux_loss=None)
#model.eval()
#model
#model(torch.zeros((1,3,224,224,)))