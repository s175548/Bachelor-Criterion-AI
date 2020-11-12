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
from object_detect.helper.evaluator import do_nms
from object_detect.get_bboxes import get_bbox_mask, create_mask_from_bbox
from object_detect.helper.generate_preds import validate
import object_detect.helper.utils as utils
import matplotlib.pyplot as plt
from object_detect.train_hpc import define_model
from data_import.tif_import import load_tif_as_numpy_array
from PIL import Image
import torchvision.transforms.functional as F


transform_function = et.ExtCompose([et.ExtCenterCrop(size=1024),
                                    et.ExtEnhanceContrast(),
                                    et.ExtToTensor()])
# et.ExtRandomCrop((256,256)), et.ExtRandomHorizontalFlip(),et.ExtRandomVerticalFlip(),
HPC = False
splitted_data = True
binary = True
tif = False

if __name__ == '__main__':

    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if HPC:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        base_path = '/zhome/dd/4/128822/Bachelorprojekt/'
        model_folder = 'faster_rcnn/'
        save_path_model = os.path.join(base_path, model_folder)
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'

        parser = argparse.ArgumentParser(description='Chooses model')
        parser.add_argument('model folder', metavar='folder', type=float, nargs='+',
                            help='model folder (three_scale, full_scale, all_bin, binary')
        # Example: model_folder = all_bin\resnet50_full_empty_0.01_all_binarySGD.pt
        model_folder = args['model folder'][0]

        model_name = 'resnet50'
        tif_path = '/zhome/db/f/128823/Bachelor/data_all_classes/tif_image'
    else:
        device = torch.device('cpu')
        model_name = 'resnet50'
        path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
        path_meta_data = r'samples/model_comparison.csv'
        optim = "SGD"
        tif_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\TIF\good_area1.png'
        save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\predictions\vda4'

    print("Device: %s" % device)
    data_loader = DataLoader(data_path=path_original_data,
                             metadata_path=path_meta_data)
    patch_size = 128

    array = load_tif_as_numpy_array(tif_path)
    split_imgs, split_x_y, patch_dimensions = data_loader.generate_tif_patches(array, patch_size=patch_size,
                                                                               padding=50, with_pad=True)

    model = define_model(num_classes=2, net=model_name, anchors=((16,), (32,), (64,), (128,), (256,)))

    if HPC:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\faster_rcnn'
        PATH = os.path.join(PATH, model_folder)

    else:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\faster_rcnn\all_bin\resnet50_full_empty_0.01_all_binarySGD.pt'
        # PATH = os.path.join(PATH, model_folder)

    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model["model_state"])
    model.to(device)
    model.eval()
    print("Model loaded and ready to be evaluated!")

    target_tif = []
    print("Loop over: ", split_x_y[0])
    for i in range(split_x_y[0]):
        print("i ", i)
        pred_stack = []
        for j in range(split_x_y[1]):
            print(j)
            label = Image.fromarray(np.zeros(split_imgs[i * split_x_y[1] + j].shape, dtype=np.uint8))
            image = Image.fromarray(split_imgs[i * split_x_y[1] + j].astype(np.uint8))
            size = image.size

            if j == 0:
                F.pad(image, padding=(0, 0, 50, 0), padding_mode='reflect')
            if j == split_x_y[1] - 1:
                F.pad(image, padding=(50, 0, 0, 0), padding_mode='reflect')
            if i == 0:
                F.pad(image, padding=(0, 50, 0, 0), padding_mode='reflect')
            if i == split_x_y[0] - 1:
                F.pad(image, padding=(0, 0, 0, 50), padding_mode='reflect')

            image, _ = transform_function(image, label)
            image = image.unsqueeze(0).to(device, dtype=torch.float32)
            output = model(list(image))
            outputs = [{k: v.to(device) for k, v in t.items()} for t in output]

            boxes = outputs[0]['boxes'].cpu()
            scores = outputs[0]['scores'].cpu()
            preds = outputs[0]['labels'].cpu()
            new_boxes, _, _ = do_nms(boxes, scores, preds, threshold=0.2)
            pred = create_mask_from_bbox(new_boxes.detach().cpu().numpy(),size)
            pred = pred[50:-50, 50:-50]
            if isinstance(pred_stack, list):
                pred_stack = pred
            else:
                pred_stack = np.hstack((pred_stack, pred))

        if isinstance(target_tif, list):
            target_tif = pred_stack
        else:
            target_tif = np.vstack((target_tif, pred_stack))

    PIL.Image.fromarray(target_tif.astype(np.uint8) * 255).save(save_path + '/pred_vda_04.png')
