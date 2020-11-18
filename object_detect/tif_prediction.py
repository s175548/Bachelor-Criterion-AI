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
from object_detect.get_bboxes import get_bbox_mask, create_mask_from_bbox, adjust_bbox_tif
from object_detect.helper.generate_preds import validate
import object_detect.helper.utils as utils
import matplotlib.pyplot as plt
from object_detect.train_hpc import define_model
from data_import.tif_import import load_tif_as_numpy_array
from PIL import Image
import torchvision.transforms.functional as F
resize = False
if resize:
    transform_function = et.ExtCompose([et.ExtResize(scale=0.5),
                                        et.ExtEnhanceContrast(),
                                        et.ExtToTensor()])
    patch_size = 512
else:
    transform_function = et.ExtCompose([et.ExtEnhanceContrast(),
                                        et.ExtToTensor()])
    patch_size = 256

# et.ExtRandomCrop((256,256)), et.ExtRandomHorizontalFlip(),et.ExtRandomVerticalFlip(),
HPC = True
splitted_data = True
binary = True
tif = True
brevetti = False

def output(model,array):
    image = Image.fromarray(array)
    size = image.size
    print("1 output size: ", size)
    image2, _ = transform_function(image, label)
    image2 = image2.unsqueeze(0).to(torch.device('cuda'), dtype=torch.float32)
    print(np.shape(image2))
    output = model(list(image2))
    return [{k: v.to(torch.device('cuda')) for k, v in t.items()} for t in output], size

if __name__ == '__main__':

    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print("So far")
    if HPC:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        cpu_device = torch.device('cpu')

        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'

        parser = argparse.ArgumentParser(description='Chooses model')
        parser.add_argument('model folder', metavar='folder', type=str, nargs='+',
                            help='model folder (three_scale, full_scale, all_bin, binary')
        args = vars(parser.parse_args())

        model_folder = args['model folder'][0]
        if brevetti:
            save_path = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/tif_brevetti'
        else:
            save_path = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/vda4'
        model_name = 'resnet50'
        tif_path = '/work3/s173934/Bachelorprojekt/tif_img'
        if brevetti:
            tif_path = os.path.join(tif_path,'RED_HALF02_grain_01_v.tif')
        else:
            tif_path = os.path.join(tif_path,'WALKNAPPA_VDA_04_grain_01_v.tif')

        if model_folder == 'all_bin':
            pt_name = 'resnet50_full_empty_0.01_all_binarySGD.pt'
            exp = 'crop_all_classes'

        if model_folder == 'binary':
            pt_name = 'resnet50_full_empty_0.01_binarySGD.pt'
            exp = 'crop_3_classes'

        if model_folder == 'three_scale':
            pt_name = 'resnet50_full_empty_0.01_binary_scaleSGD.pt'
            exp = 'resize_3_classes'

        if model_folder == 'full_scale':
            pt_name = 'resnet50_all_binary_scale_part2SGD.pt'
            exp = 'resize_all_classes'


    else:
        device = torch.device('cpu')
        cpu_device = torch.device('cpu')

        model_name = 'resnet50'
        path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
        path_meta_data = r'samples/model_comparison.csv'
        optim = "SGD"
        tif_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\TIF\good_area1.png'
        save_path = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\predictions\vda4'

    print("Device: %s" % device)
    data_loader = DataLoader(data_path=path_original_data,
                             metadata_path=path_meta_data)

    array = load_tif_as_numpy_array(tif_path)
    print("Shape array: ", np.shape(array))
    split_imgs, split_x_y, patch_dimensions = data_loader.generate_tif_patches2(array, patch_size=patch_size,
                                                                               padding=50, with_pad=True)

    model = define_model(num_classes=2, net=model_name, anchors=((16,), (32,), (64,), (128,), (256,)),box_score=0.5)

    if HPC:
        base_path = r'/zhome/dd/4/128822/Bachelorprojekt/faster_rcnn'
        PATH = os.path.join(base_path, model_folder)
        PATH = os.path.join(PATH, pt_name)
    else:
        PATH = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\last_round\faster_rcnn\all_bin\resnet50_full_empty_0.01_all_binarySGD.pt'
        # PATH = os.path.join(PATH, model_folder)

    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model["model_state"])
    model.to(device)
    model.eval()
    print("Model loaded and ready to be evaluated!")

    target_tif = []
    print("Loop over: ", split_x_y[0], "and ", split_x_y[1])
    pred_counter = 0
    for i in range(split_x_y[0]):
        print("i ", i)
        pred_stack = []
        for j in range(split_x_y[1]):
            label = Image.fromarray(np.zeros(split_imgs[i * split_x_y[1] + j].size, dtype=np.uint8))
            outputs, size = output(model,array=split_imgs[i * split_x_y[1] + j])

            boxes = outputs[0]['boxes'].cpu()
            scores = outputs[0]['scores'].cpu()
            preds = outputs[0]['labels'].cpu()
            new_boxes, new_scores, _ = do_nms(boxes.detach(), scores.detach(), preds.detach(), threshold=0.2)
            pred = create_mask_from_bbox(new_boxes.detach().cpu().numpy(),size)
            pred, num_boxes = adjust_bbox_tif(new_boxes.detach().cpu().numpy(),adjust=50,size=size[0])
            print("5 pred before [50:-50]: ", np.shape(pred))
            pred_counter += num_boxes
            pred = pred[50:-50, 50:-50]
            if isinstance(pred_stack, list):
                print("6 pred ",np.shape(pred))
                pred_stack = pred
            else:
                print("6 pred ",np.shape(pred))
                pred_stack = np.hstack((pred_stack, pred))

        if isinstance(target_tif, list):
            target_tif = pred_stack
        else:
            target_tif = np.vstack((target_tif, pred_stack))
    print("Model: ", exp)
    if brevetti:
        Image.fromarray(target_tif.astype(np.uint8)).save(save_path + '/brevetti_{}.png'.format(exp))
    else:
        Image.fromarray(target_tif.astype(np.uint8)).save(save_path + '/vda_{}.png'.format(exp))
