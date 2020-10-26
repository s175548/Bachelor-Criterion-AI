import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
import random,cv2
def randomCrop(img, mask, width, height):
    crop_img = np.empty((3,width,height))
    crop_mask = np.empty((width,height))
    x = random.randint(0, img.shape[1]-width)
    y = random.randint(0,img.shape[2]- height)
    img_num = img.numpy()
    mask_num = mask.numpy()
    for i in range(3):
        crop_img[i] = img_num[i][x:x+width,y:y+height ]
    crop_mask = mask_num[x:x+width,y:y+height ]
    return torch.from_numpy(crop_img),torch.from_numpy(crop_mask)
def pad(img,mask,size,ignore_idx):
    #        topBorderWidth,bottomBorderWidth, leftBorderWidth,  rightBorderWidth,
    img_num = img.numpy()
    mask_num = mask.numpy()
    pad_img = np.empty((3,size,size))
    pad_mask = np.empty(( size, size))

    height_border = (size-img_num.shape[1] )// 2
    width_border = (size - img_num.shape[2]) //2
    if (size-img_num.shape[1])%2 != 0:
        rest_height = 1
    else:
        rest_height = 0
    if (size-img_num.shape[2])%2 != 0:
        rest_width = 1
    else:
        rest_width = 0

    if (width_border >= 0 and height_border >= 0):
        for i in range(3):
            pad_img[i] = cv2.copyMakeBorder(img_num[i], height_border, height_border + rest_height, width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)
        pad_mask = cv2.copyMakeBorder(mask_num, height_border, height_border + rest_height, width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)
    elif height_border >= 0:
        rand_start = random.randint(0, abs(width_border) * 2)
        for i in range(3):
            pad_img[i] = cv2.copyMakeBorder(img_num[i], height_border, height_border + rest_height, 0,0, cv2.BORDER_CONSTANT, value=ignore_idx)[:,rand_start:rand_start+size]
        pad_mask = cv2.copyMakeBorder(mask_num, height_border, height_border + rest_height, 0,0, cv2.BORDER_CONSTANT, value=ignore_idx)[:,rand_start:rand_start+size]
    elif width_border >= 0:
        rand_start = random.randint(0,abs(height_border)*2)
        for i in range(3):
            pad_img[i] = cv2.copyMakeBorder(img_num[i], 0, 0 , width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)[rand_start:rand_start+size,:]
        pad_mask = cv2.copyMakeBorder(mask_num, 0, 0 , width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)[rand_start:rand_start+size,:]

    return torch.from_numpy(pad_img),torch.from_numpy(pad_mask)