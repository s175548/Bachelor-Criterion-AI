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
from object_detect.helper.generate_preds import validate
import object_detect.helper.utils as utils
import matplotlib.pyplot as plt
from object_detect.train_hpc import define_model

transform_function = et.ExtCompose([et.ExtCenterCrop(size=1024),
                                    et.ExtEnhanceContrast(),
                                    et.ExtToTensor()])
#et.ExtRandomCrop((256,256)), et.ExtRandomHorizontalFlip(),et.ExtRandomVerticalFlip(),
HPC=True
splitted_data=True
binary=True

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

        parser = argparse.ArgumentParser(description='Take parameters')
        parser.add_argument('model name', metavar='model', type=str, nargs='+',help='choose either mobilenet or resnet50')
        parser.add_argument('scale', metavar='scale', type=str, nargs='+',help='choose either resize or crop')
        parser.add_argument('dataset', metavar='dataset', type=str, nargs='+',help='choose either three or extended')
        args = vars(parser.parse_args())
        model_name = args['model name'][0]
        setup = args['scale'][0]
        classes = args['dataset'][0]
        if setup == 'resize':
            scale = True
        else:
            scale = False
        if classes == 'three':
            all_classes = False
        else:
            all_classes = True
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

        path_save = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/'
        path_save = os.path.join(path_save, save_fold)
        save_folder = os.path.join(path_save, model_name)
        save_path_exp = os.path.join(save_path_model,save_fold)
    else:
        device = torch.device('cpu')
        num_epoch = 1
        path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
        path_meta_data = r'samples/model_comparison.csv'
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

        path_save = '/Users/johan/iCloudDrive/DTU/KID/BA/Kode/FRCNN/'
        save_folder = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Predictions_FRCNN'

    print("Device: %s" % device)
    data_loader = DataLoader(data_path=path_original_data,
                                 metadata_path=path_meta_data)

    color_dict = data_loader.color_dict_binary
    target_dict = data_loader.get_target_dict()
    annotations_dict = data_loader.annotations_dict

    batch_size = 4

    file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] != "k"])
    N_files = len(file_names_train)
    file_names_train = file_names_train[file_names_train != ".DS_S"]

    file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
    N_files = len(file_names_val)

    train_dst = LeatherDataZ(path_mask=path_train, path_img=path_train, list_of_filenames=file_names_train,
                            bbox=True, multi=False,
                            transform=transform_function, color_dict=color_dict, target_dict=target_dict)
    val_dst = LeatherDataZ(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                          bbox=True, multi=False,
                          transform=transform_function, color_dict=color_dict, target_dict=target_dict)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    val_loader = data.DataLoader(
        val_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

    model = define_model(num_classes=2, net=model_name, anchors=((16,), (32,), (64,), (128,), (256,)))

    if HPC:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Models\binary\resnet50_full_empty_0.01_binarySGD.pt'
    else:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Models\binary\resnet50_full_empty_0.01_binarySGD.pt'
    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model["model_state"])
    model.to(device)
    model.eval()
    if HPC:
        save_path = path_save
    else:
        save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Predictions\binary'

    val_mAP, val_mAP2, cmatrix_val, cmatrix_val2 = validate(model=model, model_name=model_name,
                                                            data_loader=val_loader,
                                                            device=device,
                                                            path_save=save_path, bbox_type='empty',
                                                            val=True)
    print("Overall best with nms: ", val_mAP2)
    print("Overall best without nms is: ", val_mAP)
    print("Stats for no nms:")
    print("Overall best tp: ", cmatrix_val2["true_positives"], " out of ", cmatrix_val2["total_num_defects"],
          " with ",
          cmatrix_val2["false_positives"], " false positives, ", cmatrix_val2["false_negatives"],
          " false negatives and ",
          cmatrix_val2["true_negatives"], "true negatives")
    print("Stats for nms:")
    print("Overall best tp: ", cmatrix_val["true_positives"], " out of ", cmatrix_val["total_num_defects"],
          " with ",
          cmatrix_val["false_positives"], " false positives, ", cmatrix_val["false_negatives"],
          " false negatives and ",
          cmatrix_val["true_negatives"], "true negatives")
    print("Validation set contained ", cmatrix_val["good_leather"], " images with good leather and ",
          cmatrix_val["bad_leather"],
          " with bad leather")
    
    train_mAP, train_mAP2, cmatrix_train, cmatrix_train2 = validate(model=model,model_name=model_name,
                                                            data_loader=train_loader,
                                                            device=device,
                                                            path_save = save_path,bbox_type='empty',
                                                            val=False)



