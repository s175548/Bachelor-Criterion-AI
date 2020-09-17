from tqdm import tqdm
import torchvision
import numpy as np
from torchvision import datasets, transforms, utils
import argparse
import os
from torch.utils import data
from PIL import Image
import torch
from torchvision.models.vgg import vgg16
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et

def initialize_model(num_classes,backbone,out_channels):
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
    #backbone = vgg16(pretrained=True).features
    backbone.out_channels = out_channels
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    for param in model.parameters():
        param.requires_grad = False
    return model

def init_model(num_classes):
    #load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = num_classes  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
        print(score)
    return score, ret_samples

transform_function = et.ExtCompose([et.ExtTransformLabel(),et.ExtCenterCrop(512),et.ExtScale(300),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),])

class LeatherData(data.Dataset):

    def __init__(self,
                 path_mask,path_img,
                 transform=None):


        self.path_mask = path_mask
        self.path_img = path_img
        self.transform = transform


        file_names_mask=os.listdir(self.path_mask)
        file_names_img=os.listdir(self.path_img)

        self.images = [os.path.join(self.path_img, x) for x in file_names_img]
        self.masks = [os.path.join(self.path_mask, x) for x in file_names_mask]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':

    device = torch.device('cpu')
    print("Device: %s" % device)

    output_stride = 16
    save_val_results = False
    total_itrs = 10
    lr = 0.01
    lr_policy = 'step'
    step_size = 10000
    batch_size = 16
    val_batch_size = 4
    loss_type = "cross_entropy"
    weight_decay = 1e-4
    random_seed = 1
    print_interval = 10
    val_interval = 100
    vis_num_samples = 2
    enable_vis = True

    train_path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Train_mask'
    test_path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Test_mask'
    train_path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Train_img'
    test_path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Test_img'

    #model = initialize_model(num_classes=21,backbone=vgg16(pretrained=True).features,out_channels=512)
    model = init_model(num_classes=2)
    model.to(device)
    model.eval()

    train_dst = LeatherData(path_mask=test_path_mask,path_img=test_path_img,transform=transform_function)
    val_dst = LeatherData(path_mask=train_path_mask,path_img=train_path_img,transform=transform_function)
    train_loader = data.DataLoader(
       train_dst, batch_size=2, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=True, num_workers=2)
    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))
    i = 0

    path_test = r'C:/Users/johan/iCloudDrive/DTU/KID/BA/Kode/brevetti/'
    im = Image.open(path_test+'PennFudanPed/PNGImages/FudanPed00001.png').convert("RGB")
    im2 = Image.open(path_test+'PennFudanPed/PNGImages/FudanPed00002.png').convert("RGB")
    img = np.array(im)
    img2 = np.array(im2)
    image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, 0, 0], [1, 1, 1]),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = image_transform(img)
    img2 = image_transform(img2)
    imgs = [img, img2]
    #metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'
    for (images, labels) in train_loader:
        if i < 10:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            image = list(img.to(device) for img in images)
            pred = model(images)
            pr = model(imgs)
            break
        else:
            pass


#            a = Image.fromarray(images[0].mul(255).permute(1, 2, 0).byte().numpy())
#            Image.fromarray(pred[0]['boxes'].mul(255).byte().cpu().numpy())
#path_test = r'C:/Users/johan/iCloudDrive/DTU/KID/BA/Kode/brevetti/'
#    img = Image.open(path_test+'PennFudanPed/PNGImages/FudanPed00001.png').convert("RGB")
#    img2 = Image.open(path_test+'PennFudanPed/PNGImages/FudanPed00002.png').convert("RGB")
##    img = np.array(img)
 #   img2 = np.array(img2)
 #   image_transform = transforms.Compose([
 #       transforms.ToTensor(), transforms.Normalize([0, 0, 0], [1, 1, 1]),
 #       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 ##   image = image_transform(img)
  #  image2 = image_transform(img2)
  #  images = [image, image2]