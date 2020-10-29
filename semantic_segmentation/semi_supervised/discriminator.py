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