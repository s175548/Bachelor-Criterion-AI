'''
  the author is leilei
  you have so many choices: deeplab_v3 、or based vgg16 -> u-net
'''
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from semantic_segmentation.DeepLabV3.network.modeling import _segm_mobilenet

def discriminator(model='MobileNet',n_classes = 3):
    default_scope = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    model_dict={}
    if model=='DeepLab':
        model_dict[model]=deeplabv3_resnet101(pretrained=True, progress=True,num_classes=21, aux_loss=None)
        if default_scope:
            grad_check(model_dict[model])
        else:
            grad_check(model_dict[model], model_layers='All')
        model_dict[model].classifier[-1] = torch.nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
        model_dict[model].aux_classifier[-1] = torch.nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()


    if model=="MobileNet":
        model_dict[model] = _segm_mobilenet('deeplabv3', 'mobile_net', output_stride=8, num_classes=n_classes,pretrained_backbone=True)
        grad_check(model_dict[model],model_layers='All')
    # Setup visualization

    for model_name, model in model_dict.items():
        print(model_name)
        return model_dict


def grad_check(model,model_layers='Classifier'):
    if model_layers=='Classifier':
        print('Classifier only')
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(requires_grad=True)
    else:
        print('Whole model')
        for parameter in model.parameters():
            parameter.requires_grad_(requires_grad=True)


import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)