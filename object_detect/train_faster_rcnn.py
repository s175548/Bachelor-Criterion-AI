""" Script by Johannes B. Reiche, inspired by: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html """
import sys, os
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')

import torchvision, random
import pickle
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from object_detect.leather_data_hpc import LeatherDataZ
from data_import.data_loader import DataLoader
from torch.utils import data
import torch
import argparse
from object_detect.helper.FastRCNNPredictor import FastRCNNPredictor, FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from object_detect.helper.engine import train_one_epoch, evaluate
import object_detect.helper.utils as utils
import matplotlib.pyplot as plt
from object_detect.helper.generate_preds import validate


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

def define_model(num_classes, net, anchors,up_thres=0.5,low_thres=0.2,box_score=0.3,data='binary'):
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
        if data == 'tick_bite':
            anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),),
                                               aspect_ratios=((0.5, 1.0, 2.0),))
        else:
            anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
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
                           rpn_fg_iou_thresh=up_thres, rpn_bg_iou_thresh=low_thres,
                           box_roi_pool=roi_pooler, box_score_thresh=box_score)

    elif net == 'resnet50':
        resnet50 = init_model(num_classes=num_classes)
        anchor_sizes = anchors
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(
                resnet50.backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names='0',
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(resnet50.backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=rpn_anchor_generator, rpn_head = rpn_head,
                           rpn_fg_iou_thresh=up_thres, rpn_bg_iou_thresh=low_thres,
                           box_roi_pool=roi_pooler, box_score_thresh=box_score)
    return model

def save_model(model,save_path='/zhome/dd/4/128822/Bachelorprojekt/faster_rcnn/',HPC=True,model_name=None,optim_name=None,n_epochs=None, optimizer=None,best_map=None,best_score=None,best_ious=None,conf=None,losses=None,val_losses=None):
    """ save final model
    """
    if HPC:
        torch.save({
            "n_epochs": n_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_map": best_map,
            "best_map_w_score": best_score,
            "best_ious": best_ious,
            "conf_matrix": conf,
            "train_losses": losses,
            "val_losses": val_losses,
        }, save_path+model_name+optim_name+'.pt')
        print("Model saved as "+model_name+optim_name+'.pt')
    else:
        torch.save({
            "n_epochs": n_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
            "train_losses": losses,
            "val_losses": val_losses,
        }, save_path + model_name + optim_name + '.pt')

def freeze_layers(model,layers):
    params = [p for p in model.parameters() if p.requires_grad]
    if layers=='Classifier':
        params2freeze = params[:-8]
        for parameter in params2freeze:
            parameter.requires_grad_(requires_grad=False)
    elif layers=='RPN':
        params2freeze = params[:-14]
        for parameter in params2freeze:
            parameter.requires_grad_(requires_grad=False)
    else:
        pass

def plot_loss(N_epochs=None,train_loss=None,save_path=None,lr=None,optim_name=None,val_loss=None,exp_description = ''):
    plt.plot(range(N_epochs), train_loss, '-o')
    plt.title('Train Loss')
    plt.xlabel('N_epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_path, exp_description + optim_name + (str(lr)) + 'bin_train_loss.png'), format='png')
    plt.close()
    plt.plot(range(N_epochs), val_loss, '-o')
    plt.title('Validation Loss')
    plt.xlabel('N_epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_path, exp_description + optim_name + (str(lr)) + 'bin_val_loss.png'), format='png')
    plt.close()

def get_transform_fun(resized=False):
    if resized == True:
        transform_function = et.ExtCompose([et.ExtRandomCrop(size=2048),
                                            et.ExtRandomCrop(scale=0.7,size=None),
                                            et.ExtEnhanceContrast(),
                                            et.ExtRandomCrop(size=2048, pad_if_needed=True),
                                            et.ExtResize(scale=0.5),
                                            et.ExtRandomHorizontalFlip(p=0.5),
                                            et.ExtRandomCrop(size=512),
                                            et.ExtRandomVerticalFlip(p=0.5),
                                            et.ExtToTensor()])
    else:
        transform_function = et.ExtCompose([et.ExtRandomCrop(size=256),
                                            et.ExtRandomHorizontalFlip(p=0.5),
                                            et.ExtRandomVerticalFlip(p=0.5),
                                            et.ExtEnhanceContrast(),
                                            et.ExtToTensor()])
    return transform_function

#transform_function = et.ExtCompose([et.ExtScale(scale=0.7),et.ExtRandomCrop(scale=0.7),et.ExtRandomHorizontalFlip(p=0.5),et.ExtRandomVerticalFlip(p=0.5),et.ExtEnhanceContrast(),et.ExtToTensor()])
#et.ExtRandomCrop((256,256)), et.ExtRandomHorizontalFlip(),et.ExtRandomVerticalFlip(),
HPC=True
tick_bite=False
if tick_bite:
    splitted_data = False
else:
    splitted_data = True
binary=True
multi=False
load_model=True
if __name__ == '__main__':

    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if HPC:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        base_path = '/zhome/dd/4/128822/Bachelorprojekt/'
        model_folder = 'faster_rcnn/'
        save_path_model = os.path.join(base_path,model_folder)
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'

        parser = argparse.ArgumentParser(description='Take learning rate parameter')
        parser.add_argument('parameter choice', metavar='lr', type=float, nargs='+',help='a parameter for the training loop')
        parser.add_argument('model name', metavar='model', type=str, nargs='+',help='choose either mobilenet or resnet50')
        parser.add_argument('optimizer name', metavar='optim', type=str, nargs='+',help='choose either SGD, Adam or RMS')
        parser.add_argument('trained layers', metavar='layers', type=str, nargs='+',help='choose either full or classifier')
        parser.add_argument('bbox', metavar='bbox', type=str, nargs='+',help='choose either zero or empty')
        parser.add_argument('scale', metavar='scale', type=str, nargs='+',help='choose either resize or crop')
        parser.add_argument('dataset', metavar='dataset', type=str, nargs='+',help='choose either three or extended')
        args = vars(parser.parse_args())

        model_name = args['model name'][0]
        setup = args['scale'][0]
        classes = args['dataset'][0]
        if classes == 'three':
            all_classes = False
        else:
            all_classes = True
        if setup == 'resize':
            scale = True
            num_epoch = 100
        else:
            scale = False
            num_epoch = 100

        if binary:
            if scale:
                if all_classes:
                    path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train'
                    path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'
                    save_fold = 'full_scale/'
                    dataset = "all_binary_scale"
                else:
                    path_train = r'/work3/s173934/Bachelorprojekt/data_binary_vis_2_and_3_good_patches/train'
                    path_val = r'/work3/s173934/Bachelorprojekt/data_binary_vis_2_and_3_good_patches/val'
                    save_fold = 'three_scale/'
                    dataset = "binary_scale"
            else:
                if all_classes:
                    path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train'
                    path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'
                    save_fold = 'all_bin/'
                    dataset = "all_binary"
                else:
                    path_train = r'/work3/s173934/Bachelorprojekt/data_binary_vis_2_and_3_good_patches/train'
                    path_val = r'/work3/s173934/Bachelorprojekt/data_binary_vis_2_and_3_good_patches/val'
                    save_fold = 'binary/'
                    dataset = "binary"
        elif tick_bite:
            path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
            path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
            save_fold = 'tick_bite/'
            dataset = "tick_bite"
        else:
            path_train = r'/zhome/dd/4/128822/Bachelorprojekt/multi/train'
            path_val = r'/zhome/dd/4/128822/Bachelorprojekt/multi/val'
            save_fold = 'multi/'
            dataset = "multi"
        path_save = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/'
        path_save = os.path.join(path_save, save_fold)
        save_folder = os.path.join(path_save, 'part2')
        save_path_exp = os.path.join(save_path_model,save_fold)
        bbox_type = args['bbox'][0]
        lr = args['parameter choice'][0]
        optim = args['optimizer name'][0]
        layers_to_train = args['trained layers'][0]
    else:
        device = torch.device('cpu')
        lr = 0.01
        layers_to_train = 'classifier'
        num_epoch = 1
        path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
        path_meta_data = r'samples/model_comparison.csv'
        optim = "SGD"
        if binary:
            path_train= r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary\train'
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary\test'
            dataset = "binary_scale"
        elif tick_bite:
            path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\tick_bite'
            path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\tick_bite'
            dataset = "tick_bite"
        else:
            path_train = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\multi\train'
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\multi\test'
            dataset = "multi"

        path_save = '/Users/johan/iCloudDrive/DTU/KID/BA/Kode/FRCNN/'
        save_folder = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Predictions_FRCNN'

    print("Device: %s" % device)
    data_loader = DataLoader(data_path=path_original_data,
                                 metadata_path=path_meta_data)

    labels=['Piega', 'Verruca', 'Puntura insetto','Background']

    if binary:
        color_dict = data_loader.color_dict_binary
        target_dict = data_loader.get_target_dict()
        annotations_dict = data_loader.annotations_dict

    else:
        color_dict= data_loader.color_dict
        target_dict=data_loader.get_target_dict(labels)
        annotations_dict=data_loader.annotations_dict

    if tick_bite:
        batch_size = 4
        val_batch_size = 4
    else:
        if HPC:
            if scale:
                batch_size = 8
            else:
                batch_size = 8
            val_batch_size = 4
        else:
            batch_size = 8
            val_batch_size = 4

    if splitted_data:
        file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] != "k"])
        N_files = len(file_names_train)
        shuffled_index = np.random.permutation(len(file_names_train))
        file_names_train = file_names_train[shuffled_index]
        file_names_train = file_names_train[file_names_train != ".DS_S"]

        file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
        N_files = len(file_names_val)
        file_names_val = file_names_val[file_names_val != ".DS_S"]


        transform_function = get_transform_fun(resized=scale)

        if bbox_type == 'empty':
            train_dst = LeatherDataZ(path_mask=path_train, path_img=path_train, list_of_filenames=file_names_train,
                                    bbox=True, multi=multi,
                                    transform=transform_function, color_dict=color_dict, target_dict=target_dict)
            val_dst = LeatherDataZ(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                                  bbox=True, multi=multi,
                                  transform=transform_function, color_dict=color_dict, target_dict=target_dict)
        else:
            train_dst = LeatherData(path_mask=path_train, path_img=path_train, list_of_filenames=file_names_train,
                                    bbox=True, multi=multi,
                                    transform=transform_function, color_dict=color_dict, target_dict=target_dict)
            val_dst = LeatherData(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                                  bbox=True, multi=multi,
                                  transform=transform_function, color_dict=color_dict, target_dict=target_dict)
    else:
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

    if HPC:
        if load_model:
            model = define_model(num_classes=2, net=model_name,
                                 data=dataset, anchors=((16,), (32,), (64,), (128,), (256,)))
            PATH = r'/zhome/dd/4/128822/Bachelorprojekt/faster_rcnn/'
            if all_classes:
                PATH = os.path.join(PATH,'/full_scale/resnet50_full_empty_0.01_all_binary_scaleSGD.pt')
                print(PATH)
                start_epoch = 50
            else:
                PATH = os.path.join(PATH,'/three_scale/resnet50_full_empty_0.01_binary_scaleSGD.pt')
                start_epoch = 75
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state'])
            model.to(device)
            model.eval()
            print("Model loaded and ready to be evaluated!")
        else:
            model = define_model(num_classes=2, net=model_name,
                                 data=dataset, anchors=((16,), (32,), (64,), (128,), (256,)))
    else:
        model_names = ['mobilenet', 'resnet50']
        model_name = model_names[0]
        model = define_model(num_classes=2, net=model_name, data=dataset,anchors=((8,), (16,), (32,), (64,), (128,)))
    model.to(device)
    print("Model: ", model_name)
    print("Learning rate: ", lr)
    print("Optimizer: ", optim)
    print("Number of epochs: ", num_epoch)
    print("Trained network: ", layers_to_train)
    print("Bounding box: ", bbox_type)

    # construct an optimizer
    layers = ['Classifier', 'RPN', 'All']
    params = [p for p in model.parameters() if p.requires_grad]

    if layers_to_train == 'full':
        freeze_layers(model, layers=layers[2])
        print("Layers trained: ", layers[2])
    else:
        freeze_layers(model, layers=layers[1])
        print("Layers trained: ", layers[0], " + ", layers[1])

    params2train = [p for p in model.parameters() if p.requires_grad]
    weight_decay = 0.0001

    # Set up optimizer
    if optim == 'SGD':
        optimizer = torch.optim.SGD(params=params2train, lr=lr, momentum=0.9, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    loss_train = checkpoint['train_losses']
    loss_val = checkpoint['val_losses']
    best_map = checkpoint['best_map']
    best_map2 = 0
    best_scores = checkpoint['best_scores']
    best_score = best_scores[0]
    best_ious = checkpoint['best_ious']
    best_iou = best_ious[0]
    cmatrix = checkpoint['conf_matrix']

    val_boxes = []
    val_targets = []
    print("About to train")
    best_epoch = 0
    for epoch in range(start_epoch,100):
        curr_loss_train = []
        curr_loss_val = []
        # train for one epoch, printing every 10 iterations
        model, loss, _, _ = train_one_epoch(model, model_name, optim_name=optim, lr=lr, layers=layers_to_train,
                                            optimizer=optimizer,
                                            data_loader=train_loader, device=device, epoch=epoch+1,print_freq=20,
                                                    loss_list=curr_loss_train,save_folder=save_folder)
        loss_train.append(loss)

        # evaluate on the test dataset
        mAP, mAP2, val_loss, vbox_p, vbox, conf, _, mIoU = evaluate(model, model_name, optim_name=optim, lr=lr,
                                                                        layers=layers_to_train,
                                                                        data_loader=val_loader,
                                                                        device=device,N=epoch+1,
                                                                        loss_list=curr_loss_val,
                                                                        save_folder=save_folder,
                                                                        risk=risk,multi=multi,scale=scale)
        if mAP > best_score:  # save best model
            best_score = mAP
            best_scores.append(best_score)
            best_scores.sort(reverse=True)
            best_scores = best_scores[:5]
        elif mAP > min(best_scores):
            best_scores.append(mAP)
            best_scores.sort(reverse=True)
            best_scores = best_scores[:5]
        if mIoU > best_iou:
            best_iou = mIoU
            best_ious.append(best_iou)
            best_ious.sort(reverse=True)
            best_ious = best_ious[:5]
        elif mIoU > min(best_ious):
            best_ious.append(mIoU)
            best_ious.sort(reverse=True)
            best_ious = best_ious[:5]

        loss_val.append(val_loss)
        val_boxes.append(vbox_p)
        val_targets.append(vbox)
        if mAP > best_map:
            best_map = mAP
            if HPC:
                best_model = define_model(num_classes=2, net=model_name,
                                 data=dataset, anchors=((16,), (32,), (64,), (128,), (256,)))
                best_model.load_state_dict(model.state_dict())
                best_model.to(device)
                best_epoch = epoch

            cmatrix["num_defects"] = conf["total_num_defects"]
            cmatrix["img_bad"] = conf["bad_leather"]
            cmatrix["img_good"] = conf["good_leather"]
            ################################################
            cmatrix["highest_tp"] = conf["true_positives"]
            cmatrix["lowest_fp"] = conf["false_positives"]
            cmatrix["lowest_fn"] = conf["false_negatives"]
            cmatrix["highest_tn"] = conf["true_negatives"]
        if mAP2 > best_map2:
            best_map2 = mAP2
    print("Average nr. of predicted boxes: ", val_boxes[-1], " model = ", model_name, "lr = ", lr)
    print("Actual average nr. of boxes: ", val_targets[-1])
    print("Overall best with nms: ", best_map, " for learning rate: ", lr, "model ", model_name, "layers ", layers_to_train, "epoch ", best_epoch)
    print("Overall best without nms is: ", best_map2, " for learning rate: ", lr, "model ", model_name)
    print("Dataset: ", dataset)
    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))
    print("Bbox_type: ", bbox_type)
    print("Stats for nms")
    print("Overall best tp: ", cmatrix["highest_tp"], " out of ", cmatrix["num_defects"], " with ", cmatrix["lowest_fp"], " false positives, ", cmatrix["lowest_fn"], " false negatives and ", cmatrix["highest_tn"], "true negatives")
    print("Validation set contained ", cmatrix["img_good"]," images with good leather and ", cmatrix["img_bad"], " with bad leather")
    print("Top 5 mAP with nms: ", best_scores)
    print("Best mean IoU of defects with nms: ", best_iou)
    print("Top 5 best mean IoU of defects with nms: ", best_ious)


    if HPC:
        save_model(model=best_model, save_path=os.path.join(save_path_model,save_fold),HPC=HPC,
                   model_name="{}_{}_{}".format(model_name, dataset, 'part2'), optim_name=optim,
                   n_epochs=best_epoch, optimizer=optimizer,
                   best_map=best_map, best_score=best_scores, best_ious=best_ious, conf=conf, losses=loss_train, val_losses=loss_val)
        best_model.eval()
        _,_,_,_ = validate(model=best_model, model_name=model_name,
                           data_loader=val_loader, device=device,
                           path_save=save_folder,bbox_type=bbox_type,
                           val=True,bbox=False)
        _,_,_,_ = validate(model=best_model, model_name=model_name,
                           data_loader=train_loader, device=device,
                           path_save=save_folder,bbox_type=bbox_type,
                           val=False,bbox=False)
        plot_loss(N_epochs=num_epoch,train_loss=loss_train,save_path=save_path_exp,lr=lr,optim_name=optim,
                  val_loss=loss_val,exp_description=model_name+'part2')