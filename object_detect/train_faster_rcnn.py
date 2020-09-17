""" Script by Johannes B. Reiche, inspired by: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html """
import torchvision
from torch.utils import data
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from object_detect.helper.engine import train_one_epoch, evaluate

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


transform_function = et.ExtCompose([et.ExtTransformLabel(),et.ExtCenterCrop(512),et.ExtScale(100),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),])

if __name__ == '__main__':

    device = torch.device('cpu')
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % device)

    model = init_model(num_classes=2)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epoch = 1
    print_freq = 10

    train_path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Train_mask'
    test_path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Test_mask'
    train_path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Train_img'
    test_path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Test_img'

    batch_size = 16
    val_batch_size = 4

    # Define dataloaders
    train_dst = LeatherData(path_mask=train_path_mask,path_img=train_path_img,transform=transform_function)
    val_dst = LeatherData(path_mask=test_path_mask,path_img=test_path_img,transform=transform_function)

    train_loader = data.DataLoader(
       train_dst, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=True, num_workers=2)
    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))

    for epoch in range(num_epoch):
        print("About to train")
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=print_freq)
        # update the learning rate
        lr_scheduler.step()
        print("\n Finished training for epoch!")
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        print("\n Finished evaluation for epoch!")