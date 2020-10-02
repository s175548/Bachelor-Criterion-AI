""" Script by Johannes B. Reiche, inspired by: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html """
import sys, os
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')

import torchvision, random
from torch.utils import data
import pickle
import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from object_detect.leather_data_hpc import LeatherData
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
    }, '/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI/faster_rcnn/'+model_name+'.pt')
    print("Model saved as "+model_name+'.pt')

transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtRandomCrop((256)),et.ExtToTensor()])

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % device)

    learning_rates = [0.05, 0.005, 0.0005]

    path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_28_09/mask'
    path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_28_09/img'

    batch_size = 8
    val_batch_size = 8
    num_epoch = 1

    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[:-4] != ".DS_S"])
    N_files = len(file_names)
    shuffled_index = np.random.permutation(len(file_names))
    file_names_img = file_names[shuffled_index]
    file_names = file_names[file_names != ".DS_S"]

    # Define dataloaders
    train_dst = LeatherData(path_mask=path_mask, path_img=path_img,list_of_filenames=file_names[:round(N_files * 0.80)],
                            transform=transform_function)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img,list_of_filenames=file_names[round(N_files * 0.80):],
                          transform=transform_function)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

    for lr in learning_rates:
        model = init_model(num_classes=2)
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        #params_to_train = params[64:]
        optimizer = torch.optim.SGD(params, lr=lr,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=5,
                                                       gamma=0.5)

        loss_train = []
        risk = True
        best_map = 0
        for epoch in range(num_epoch):
            print("About to train")
            curr_loss_train = []
            # train for one epoch, printing every 10 iterations
            model, loss, _, _ = train_one_epoch2(model, optimizer, train_loader, device, epoch,print_freq=5,
                                                        loss_list=curr_loss_train,risk=risk)
            loss_train.append(loss)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            mAP, vbox_p, vbox = evaluate(model, val_loader, device=device,N=epoch,risk=risk)

            checkpoint = mAP
            if checkpoint > best_map:
                best_map = checkpoint

        save_model(model,"{}".format(lr),n_epochs=num_epoch,optimizer=optimizer,
                   scheduler=lr_scheduler,best_score=best_map,losses=loss_train)
