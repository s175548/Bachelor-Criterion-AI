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

#transform_function = et.ExtCompose([et.ExtCenterCrop(size=1024),
#                                    et.ExtEnhanceContrast(),
#                                    et.ExtToTensor()])
transform_function = et.ExtCompose([et.ExtRandomCrop(size=2048),
                                    et.ExtRandomCrop(scale=0.7, size=None),
                                    et.ExtEnhanceContrast(),
                                    et.ExtRandomCrop(size=2048, pad_if_needed=True),
                                    et.ExtResize(scale=0.5),
                                    et.ExtRandomHorizontalFlip(p=0.5),
                                    et.ExtRandomCrop(size=512),
                                    et.ExtRandomVerticalFlip(p=0.5),
                                    et.ExtToTensor()])
#et.ExtRandomCrop((256,256)), et.ExtRandomHorizontalFlip(),et.ExtRandomVerticalFlip(),
HPC=False
splitted_data=True
binary=True
tif=False

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

        parser = argparse.ArgumentParser(description='Chooses model')
        parser.add_argument('model folder', metavar='folder', type=float, nargs='+',
                            help='model folder (three_scale, full_scale, all_bin, binary')
        # Example: model_folder = all_bin\resnet50_full_empty_0.01_all_binarySGD.pt
        model_folder = args['model folder'][0]

        model_name = 'resnet50'
        path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train'
        path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'
        save_fold = 'full_scale/'
        dataset = "all_binary_scale"
        path_save = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/test'
    else:
        device = torch.device('cpu')
        model_name = 'resnet50'
        path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
        path_meta_data = r'samples/model_comparison.csv'
        optim = "SGD"
        if binary:
            path_train= r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary\train'
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary\test'
            dataset = "binary_scale"

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
    val_dst = LeatherDataZ(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val[:10],
                          bbox=True, multi=False,
                          transform=transform_function, color_dict=color_dict, target_dict=target_dict)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    val_loader = data.DataLoader(
        val_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

    model = define_model(num_classes=2, net=model_name, anchors=((16,), (32,), (64,), (128,), (256,)))

    if HPC:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\faster_rcnn'
        PATH = os.path.join(PATH, model_folder)

    else:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\faster_rcnn\all_bin\resnet50_full_empty_0.01_all_binarySGD.pt'
        #PATH = os.path.join(PATH, model_folder)

    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model["model_state"])
    model.to(device)
    model.eval()
    print("Model loaded and ready to be evaluated!")
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
    print("Stats for nms:")
    print("Overall best tp: ", cmatrix_val["true_positives"], " out of ", cmatrix_val["total_num_defects"],
          " with ",
          cmatrix_val["false_positives"], " false positives, ", cmatrix_val["false_negatives"],
          " false negatives and ",
          cmatrix_val["true_negatives"], "true negatives")
    print("Validation set contained ", cmatrix_val["good_leather"], " images with good leather and ",
          cmatrix_val["bad_leather"],
          " with bad leather")
    
    #train_mAP, train_mAP2, cmatrix_train, cmatrix_train2 = validate(model=model,model_name=model_name,
    #                                                        data_loader=train_loader,
    #                                                        device=device,
    #                                                        path_save = save_path,bbox_type='empty',
    #                                                        val=False)



