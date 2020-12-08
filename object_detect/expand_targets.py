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
from torchvision.models.segmentation import deeplabv3_resnet101


def get_predictions(samples,ids,path_save,file_names):
    for (img, m), id in zip(samples, ids):
        image = (img[i].detach().cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
        mask = m[i]
        for i in range(len(ids)):
            Image.fromarray(image.astype(np.uint8)).save(
                path_save + '/{}_img.png'.format(file_names[ids[i].numpy()[0]]),format='PNG')
            Image.fromarray(mask.astype(np.uint8)).save(
                path_save + '/{}_mask.png'.format(file_names[ids[i].numpy()[0]]), format='PNG')

transform_function = et.ExtCompose([et.ExtToTensor()])

HPC=False
splitted_data=True
binary=True
tif=False

if __name__ == '__main__':

    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device('cpu')
    model_name = 'resnet50'
    path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
    path_meta_data = r'samples/model_comparison.csv'
    optim = "SGD"
    resize = True
    #path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\validation\val_all_class_crop'
    #save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\expand\crop'

    path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\validation\val_all_class_resize'
    save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\expand\resize'

    data_loader = DataLoader(data_path=path_original_data,
                                 metadata_path=path_meta_data)

    color_dict = data_loader.color_dict_binary
    target_dict = data_loader.get_target_dict()
    annotations_dict = data_loader.annotations_dict

    batch_size = 8

    file_names_val = np.array([image_name[:-8] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
    file_names_val = file_names_val[file_names_val != ".DS_S"]
    N_files = len(file_names_val)

    val_dst = LeatherDataZ(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                          bbox=True, multi=False,
                          transform=transform_function, color_dict=color_dict, target_dict=target_dict)

    val_loader = data.DataLoader(
        val_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Val set: %d" %(len(val_dst)))

    for (image, labels, masks) in data_loader:
        images = list(img.to(device) for img in image)
        targets = list({k: v.to(device, dtype=torch.long) for k, v in t.items()} for t in labels)

        ids = [targets[i]['image_id'].cpu() for i in range(len(targets))]
        expanded_targets = []

        for j in range(len(ids)):
            etar = expanded_targets(targets[j]['boxes'], masks[j], targets[j]['labels'], expand=expand)
            expanded_targets.append(etar)

        samples = []
        samples.append((images, expanded_targets))
        get_predictions(samples, model_name, ids, path_save=save_path, file_names=file_names, val=val)


