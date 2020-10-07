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
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from object_detect.helper.engine import train_one_epoch2, evaluate
import object_detect.helper.utils as utils

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


def define_model(num_classes, net, anchors):
    if net == 'mobilenet':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280

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
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)

    elif net == 'resnet50':
        resnet50 = init_model(num_classes=num_classes)
        anchor_sizes = anchors
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        rpn_head = RPNHead(
                resnet50.backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names='0',
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(resnet50.backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=rpn_anchor_generator, rpn_head = rpn_head,
                           box_roi_pool=roi_pooler)
    return model

def save_model(model,model_name=None,n_epochs=None, optimizer=None,scheduler=None,best_score=None,losses=None):
    """ save final model
    """
    torch.save({
        "n_epochs": n_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "train_losses": losses,
    }, '/zhome/dd/4/128822/Bachelorprojekt/faster_rcnn/'+model_name+'.pt')
    print("Model saved as "+model_name+'.pt')

#transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtRandomCrop((512)),et.ExtToTensor()])
transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtToTensor()])
binary=True
tick_bite=False
multi=False
load_model=False
if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % device)

    #learning_rates = [0.01, 0.001, 0.0001]
    learning_rates = [0.001]

    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
    path_meta_data = r'samples/model_comparison.csv'

    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    #labels=['Piega', 'Verruca', 'Puntura insetto','Background']
    # path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
    # path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
    if binary:
        path_mask = r'/zhome/dd/4/128822/Bachelorprojekt/binary'
        path_img = r'/zhome/dd/4/128822/Bachelorprojekt/binary'

    elif tick_bite:
        path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
        path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
    else:
        path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_28_09/mask'
        path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_28_09/img'
        #path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_multi'
        #path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_multi'

    data_loader = DataLoader(data_path=path_original_data,
                                 metadata_path=path_meta_data)
    color_dict = data_loader.color_dict_binary
    target_dict = data_loader.get_target_dict()
    annotations_dict = data_loader.annotations_dict
    batch_size = 4
    val_batch_size = 4
    num_epoch = 100

    file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[-5] != 'k'])
    N_files = len(file_names)
    shuffled_index = np.random.permutation(len(file_names))
    file_names_img = file_names[shuffled_index]

    train_dst = LeatherData(path_mask=path_mask, path_img=path_img,
                            list_of_filenames=file_names[:round(N_files * 0.80)],
                            bbox=True,
                            transform=transform_function, color_dict=color_dict, target_dict=target_dict)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img, list_of_filenames=file_names[round(N_files * 0.80):],
                          bbox=True,
                          transform=transform_function, color_dict=color_dict, target_dict=target_dict)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

    overall_best = 0
    overall_best2 = 0
    best_lr = 0
    best_lr2 = 0
    model_names = ['mobilenet', 'resnet50']

    if load_model == True:

    for lr in learning_rates:
        model_name = model_names[0]
        model = define_model(num_classes=2,net=model_name)
        model.to(device)
        print("Model: ", model_name)
        print("Learning rate: ", lr)
        if load_model == True:
            model.l
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        #params_to_train = params[64:]
        optimizer = torch.optim.SGD(params, lr=lr,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=50,
                                                       gamma=0.5)

        loss_train = []
        risk = True
        best_map = 0
        best_map2 = 0
        avg_pred_box = []
        avg_target_box = []
        for epoch in range(num_epoch):
            print("About to train")
            curr_loss_train = []
            # train for one epoch, printing every 10 iterations
            model, loss, pred_box, target_box = train_one_epoch2(model, model_name, optimizer, train_loader, device, epoch=epoch+1,print_freq=20,
                                                        loss_list=curr_loss_train,risk=risk)
            loss_train.append(loss)
            avg_pred_box.append(pred_box)
            avg_target_box.append(target_box)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            mAP, mAP2, vbox_p, vbox = evaluate(model, model_name, val_loader, device=device,N=epoch+1,risk=risk)

            checkpoint = mAP
            if checkpoint > best_map:
                best_map = checkpoint
                print("Best mAP: ", best_map," epoch nr. : ", epoch+1, "model: ", model_name, "lr: ", lr)
                print("Average nr. of predicted boxes: ", avg_pred_box[-1])
                print("Actual average nr. of boxes: ", avg_target_box[-1])
            if mAP2 > best_map2:
                best_map2 = mAP2
                print("Best mAP with scores: ", best_map2," epoch nr. : ", epoch+1, "model: ", model_name, "lr: ", lr)
        #save_model(model, "{}_{}".format(model_name, lr), n_epochs=num_epoch, optimizer=optimizer,
        #           scheduler=lr_scheduler, best_score=best_map, losses=loss_train)
        print("Average nr. of predicted boxes: ", avg_pred_box[-1], " model = ", model_name, "lr = ", lr)
        print("Actual average nr. of boxes: ", avg_target_box[-1])
    if overall_best < best_map:
        overall_best = best_map
        best_lr = lr
    print("Overall best is: ", overall_best, " for learning rate: ", best_lr, "model ", model_name)
    if overall_best2 < best_map2:
        overall_best2 = best_map2
        best_lr2 = lr
    print("Overall best with scores is: ", overall_best2, " for learning rate: ", best_lr2, "model ", model_name)

