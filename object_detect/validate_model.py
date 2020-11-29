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


#transform_function_C = et.ExtCompose([et.ExtRandomCrop(size=256),
#                                    et.ExtEnhanceContrast(),
#                                    et.ExtToTensor()])
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
        exp_name = 'three_scale'

        if exp_name == 'all_bin':
            pt_name = 'resnet50_full_empty_0.01_all_binarySGD.pt'
            exp = 'crop_all_classes'
            save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Predictions\all_bin'
            resize = False
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\validation\val_all_class_crop'

        if exp_name == 'binary':
            pt_name = 'resnet50_full_empty_0.01_binarySGD.pt'
            exp = 'crop_3_classes'
            save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Predictions\binary'
            resize = False
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\validation\val_3_class_crop'


        if exp_name == 'three_scale':
            pt_name = 'resnet50_full_empty_0.01_binary_scaleSGD.pt'
            exp = 'resize_3_classes'
            save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Predictions\three_scale'
            resize = True
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\validation\val_3_class_resize'


        if exp_name == 'full_scale':
            pt_name = 'resnet50_all_binary_scale_part2SGD.pt'
            exp = 'resize_all_classes'
            save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Predictions\full_scale'
            resize = True
            path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\validation\val_all_class_resize'



    print("Device: %s" % device)
    print("Exp: ", exp_name)
    data_loader = DataLoader(data_path=path_original_data,
                                 metadata_path=path_meta_data)

    color_dict = data_loader.color_dict_binary
    target_dict = data_loader.get_target_dict()
    annotations_dict = data_loader.annotations_dict

    batch_size = 8

    file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
    N_files = len(file_names_val)

    val_dst = LeatherDataZ(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                          bbox=True, multi=False,
                          transform=transform_function, color_dict=color_dict, target_dict=target_dict)

    val_loader = data.DataLoader(
        val_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Val set: %d" %(len(val_dst)))

    model = define_model(num_classes=2, net=model_name, anchors=((16,), (32,), (64,), (128,), (256,)),box_score=0.6)

    if HPC:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\faster_rcnn'
        PATH = os.path.join(PATH, model_folder)

    else:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\faster_rcnn'
        PATH = os.path.join(PATH, exp_name)
        PATH = os.path.join(PATH, pt_name)
        #PATH = os.path.join(PATH, model_folder)

    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model["model_state"])
    model.to(device)
    model.eval()

    print("Model loaded and ready to be evaluated!")
    if HPC:
        save_path = path_save

    val_mAP, val_mAP2, cmatrix_val, cmatrix_val2 = validate(model=model, model_name=model_name,
                                                            data_loader=val_loader,
                                                            device=device,
                                                            path_save=save_path, bbox_type='empty',
                                                            file_names=file_names_val,
                                                            resize=resize,
                                                            val=True)
    print("Experiment: ", exp)
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



