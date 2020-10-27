""" Script by Johannes B. Reiche, inspired by: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html """
import sys, os
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')

import torchvision, random
import pickle
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
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
from object_detect.train_hpc import define_model



transform_function = et.ExtCompose([et.ExtRandomCrop(scale=0.7),et.ExtRandomHorizontalFlip(p=0.5),et.ExtRandomVerticalFlip(p=0.5),et.ExtEnhanceContrast(),et.ExtToTensor()])
#et.ExtRandomCrop((256,256)), et.ExtRandomHorizontalFlip(),et.ExtRandomVerticalFlip(),
tick_bite=False
if tick_bite:
    splitted_data = False
else:
    splitted_data = True
binary=True
scale=True
multi=False

if __name__ == '__main__':

    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device('cpu')
    num_epoch = 1
    path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
    path_meta_data = r'samples/model_comparison.csv'
    if binary:
        if scale:
            path_train = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\full_scale\train'
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\full_scale\val'
            dataset = "binary_scale"
        else:
            path_train= r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary\train'
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary\test'
            dataset = "binary"
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
        batch_size = 1
        val_batch_size = 1

    file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] != "k"])
    N_files = len(file_names_train)
    shuffled_index = np.random.permutation(len(file_names_train))
    file_names_train = file_names_train[shuffled_index]
    file_names_train = file_names_train[file_names_train != ".DS_S"]

    file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
    N_files = len(file_names_val)

    train_dst = LeatherData(path_mask=path_train, path_img=path_train, list_of_filenames=file_names_train,
                            bbox=True, multi=multi,
                            transform=transform_function, color_dict=color_dict, target_dict=target_dict)
    val_dst = LeatherData(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                          bbox=True, multi=multi,
                          transform=transform_function, color_dict=color_dict, target_dict=target_dict)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

    model_names = ['mobilenet', 'resnet50']
    model_name = model_names[0]
    model = define_model(num_classes=2, net=model_name, data=dataset,anchors=((32,), (64,), (128,), (256,), (512,)))
    PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Models\binary\mobilenet_binary_SGD.pt'
    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model["model_state"])
    model.to(device)
    model.eval()
