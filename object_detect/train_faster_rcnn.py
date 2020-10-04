""" Script by Johannes B. Reiche, inspired by: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html """
import sys, os
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')

import torchvision, random
import os, pickle
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
from torch.utils import data
import torch
from object_detect.helper.FastRCNNPredictor import FastRCNNPredictor, FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from object_detect.helper.engine3 import train_one_epoch, evaluate
import object_detect.helper.utils as utils

def init_model(num_classes):
    #load a model pre-trained pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = num_classes  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def define_model(num_classes,net):
    if net == 'mobilenet':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280
    elif net == 'resnet':
        backbone = torchvision.models.resnet50(pretrained=True)
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For resnet50, it's 256
        # so we need to add it here
        backbone.out_channels = 256

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios>
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names='0',
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,min_size=180,max_size=360,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

def save_model(model,model_name=None,n_epochs=None, optimizer=None,scheduler=None,best_score=None,losses=None,hpc=False):
    """ save final model
    """
    if hpc==False:
        torch.save({
            "n_epochs": n_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "train_losses": losses,
        }, '/Users/johan/iCloudDrive/DTU/KID/BA/Kode/FRCNN/'+model_name+'.pt')
    if hpc==True:
        torch.save({
            "n_epochs": n_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "train_losses": losses,
        }, '/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI/faster_rcnn/' + model_name + '.pt')
    print("Model saved as "+model_name+'.pt')

transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtRandomCrop((200)),et.ExtToTensor()])

HPC =False
binary=True

if __name__ == '__main__':
    if HPC:
        if binary:
            path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
            path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
        else:
            path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_multi'
            path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_multi'
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'
        num_epoch = 1
        print_freq = 20
        batch_size = 8
        val_batch_size = 8
        file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[-5] != "k"])
        N_files = len(file_names)
        split = round(N_files * 0.80)
        split_val = round(N_files * 0.80)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    else:
        path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\mask'
        path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\img'

        file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[-5] != "k"])
        N_files = len(file_names)
        num_epoch = 1
        print_freq = 10

        batch_size = 10
        val_batch_size = 5

        split = round(N_files * 0.8)
        split_val = round(N_files * 0.8)

        device = torch.device('cpu')

    print("Device: %s" % device)
    model = define_model(num_classes=2,net='mobilenet')
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #params_to_train = params[64:]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.1)

    data_loader = DataLoader(data_path=r'C:\Users\johan\OneDrive\Skrivebord\leather_patches',
                         metadata_path=r'samples\model_comparison.csv')

    labels=['Piega', 'Verruca', 'Puntura insetto','Background']


    #torch.manual_seed(2)
    #np.random.seed(2)
    #random.seed(2)

    #shuffled_index=np.random.permutation(len(file_names))
    #file_names_img=file_names[shuffled_index]
    #file_names=file_names[file_names != ".DS_S"]

    if binary:
        color_dict = data_loader.color_dict_binary
        target_dict = data_loader.get_target_dict()
        annotations_dict = data_loader.annotations_dict

    else:
        color_dict= data_loader.color_dict
        target_dict=data_loader.get_target_dict(labels)
        annotations_dict=data_loader.annotations_dict

    #scale = 512
    # Define dataloaders
    train_dst = LeatherData(path_mask=path_mask,path_img=path_img,list_of_filenames=file_names[245:split],
                            bbox=True,
                            transform=transform_function,color_dict=color_dict,target_dict=target_dict)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img,list_of_filenames=file_names[split_val:],
                          bbox=True,
                          transform=transform_function,color_dict=color_dict,target_dict=target_dict)

    train_loader = data.DataLoader(
       train_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))

    loss_train = []
    risk = False
    best_map = 0
    for epoch in range(num_epoch):
        print("About to train")
        curr_loss_train = []
        # train for one epoch, printing every 10 iterations
        model, loss, _, _ = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5,
                                             loss_list=curr_loss_train, risk=risk)
        loss_train.append(loss)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        mAP, vbox_p, vbox = evaluate(model, val_loader, device=device, N=epoch, risk=risk)

        checkpoint = mAP
        if checkpoint > best_map:
            best_map = checkpoint
        print("Best mAP for epoch nr. {} : ".format(epoch), best_map)
