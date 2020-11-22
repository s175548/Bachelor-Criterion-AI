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
from semantic_segmentation.DeepLabV3.performance_metric_function import error_count
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics
from PIL import Image


def test_tif(pred,exp,brevetti=False,resize=False):
    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
    path_meta_data = r'samples/model_comparison.csv'
    if brevetti:
        path = r'/work3/s173934/Bachelorprojekt/tif_img/annotations_RED_HALF02_grain_01_v.tif.json'
        pred = Image.open(pred)
        target = Image.open('/work3/s173934/Bachelorprojekt/tif_img/RED_HALF02_grain_01_v_target_1d.png')
        save_path = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/test/rh'
    else:
        path = r'/work3/s173934/Bachelorprojekt/tif_img/VDA4_grain_01_whole_tif.json'
        pred = Image.open(pred)
        target = Image.open('/work3/s173934/Bachelorprojekt/tif_img/WALKNAPPA_VDA_04_grain_01_target_1d.png')
        save_path = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/test/vda'

    pred = np.array(pred) / 255
    pred = pred.astype(np.uint8)
    if resize:
        target = target.resize((int(0.5 * target.size[0]), int(0.5 * target.size[1])))

    target = np.array(target, dtype=np.uint8)[:pred.shape[0], :pred.shape[1]]
    index = target == 53
    target[index] = 0
    pred[index] = 0

    data_loader = DataLoader(data_path=path_original_data,
                             metadata_path=path_meta_data)
    color_dict = data_loader.color_dict_binary
    target_dict = data_loader.get_target_dict()
    annotations_dict = data_loader.annotations_dict

    labels = ['02', 'Abassamento', 'Abbassamento', 'Area Punture insetti', 'Area aperta', 'Area vene', 'Buco',
              'Cicatrice',
              'Cicatrice aperta', 'Contaminazione', 'Crease', 'Difetto di lavorazione', 'Dirt', 'Fianco',
              'Fiore marcio',
              'Insect bite', 'Marchio', 'Microcut', 'Piega', 'Pinza', 'Pinze', 'Poro', "Puntura d'insetto",
              'Puntura insetto', 'Ruga', 'Rughe', 'Scopertura', 'Scratch', 'Smagliatura', 'Soffiatura', 'Struttura',
              'Taglio', 'Vena', 'Vene', 'Verruca', 'Wart', 'Zona aperta', 'verruca']

    metrics = [StreamSegMetrics(2), StreamSegMetrics(2), StreamSegMetrics(2)]
    false_positives = 0
    true_negatives = [0, 0]
    errors = np.array([[0, 0], [0, 0]])
    errors, false_positives, metric, target_color, pred_color, true_negatives = error_count(None,
                                                                                            pred,
                                                                                            target, data_loader,
                                                                                            labels, errors,
                                                                                            false_positives,
                                                                                            true_negatives,
                                                                                            metrics, resize=resize,
                                                                                            size=None,
                                                                                            scale=0.5, centercrop=False,
                                                                                            path=path)
    if brevetti:
        Image.fromarray(pred_color.astype(np.uint8)).save(
            save_path + r'/RH_{}_pred_color.png'.format(exp), format='PNG')
        Image.fromarray(target_color.astype(np.uint8)).save(
            save_path + r'/RH_{}_mask_color.png'.format(exp), format='PNG')
    else:
        Image.fromarray(pred_color.astype(np.uint8)).save(
            save_path + r'/VDA_{}_pred_color.png'.format(exp), format='PNG')
        Image.fromarray(target_color.astype(np.uint8)).save(
            save_path + r'/VDA_{}_mask_color.png'.format(exp), format='PNG')

    labels = ['Insect bite', 'Binary', 'Good Area']
    new_list = [
        label + '\n' + '\n'.join([f"{name}, {performance}" for name, performance in metric[i].get_results().items()])
        for
        i, label in enumerate(labels)]
    string = '\n\n'.join(
        new_list) + f'\n\nBinary: {errors[0]} \nInsect Bite: {errors[1]} \nFalse positives: {false_positives}'
    if brevetti:
        f = open(os.path.join(save_path, 'performance_rh_{}'.format(exp)), 'w')
    else:
        f = open(os.path.join(save_path, 'performance_vda_{}'.format(exp)), 'w')
    f.write(string)

    img_list = [target_color.astype(np.uint8), pred_color.astype(np.uint8)]
    img_name_list = [r'_mask_color_back.png', r'_pred_color_back.png']

    for i, mask in enumerate(img_list):
        mask_3d = mask
        label = 'Background'
        color_map_dict = data_loader.color_dict_binary
        color_id = data_loader.annotations_dict[label]
        color_map = color_map_dict[color_id]
        mask_3d[index, :] = (mask_3d[index, :] + 1) * color_map
        mask_3d = Image.fromarray(mask_3d.astype(np.uint8)).resize(
            (int(mask.shape[1] * 0.1), int(mask.shape[0] * 0.1)))
        if brevetti:
            mask_3d.save(save_path + r'/RH_{}'.format(exp) + img_name_list[i])
        else:
            mask_3d.save(save_path + r'/VDA_{}'.format(exp) + img_name_list[i])

# path = r'/work3/s173934/Bachelorprojekt/tif_img/annotations_RED_HALF02_grain_01_v.tif.json'
# pred = Image.open(r'/zhome/dd/4/128822/Bachelorprojekt/predictions/tif_brevetti/brevetti_resize_3_classes.png')
# target = Image.open('/work3/s173934/Bachelorprojekt/tif_img/RED_HALF02_grain_01_v_target_1d.png')

HPC=True
splitted_data=True
binary=True
tif=True
brevetti=True
resize=False

if __name__ == '__main__':

    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if HPC:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'
        save_path = r'/zhome/dd/4/128822/Bachelorprojekt/predictions/test'
        if brevetti:
            path = r'/work3/s173934/Bachelorprojekt/tif_img/annotations_RED_HALF02_grain_01_v.tif.json'
            pred = Image.open(r'/zhome/dd/4/128822/Bachelorprojekt/predictions/tif_brevetti/brevetti_crop_all_classes_back.png')
            target = Image.open('/work3/s173934/Bachelorprojekt/tif_img/RED_HALF02_grain_01_v_target_1d.png')
        else:
            path = r'/work3/s173934/Bachelorprojekt/tif_img/VDA4_grain_01_whole_tif.json'
            pred = Image.open(r'/zhome/dd/4/128822/Bachelorprojekt/predictions/tif_brevetti/brevetti_resize_3_classes.png')
            target = Image.open('/work3/s173934/Bachelorprojekt/tif_img/RED_HALF02_grain_01_v_target_1d.png')
            pass
    else:
        device = torch.device('cpu')
        model_name = 'resnet50'
        path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
        path_meta_data = r'samples/model_comparison.csv'


    pred = np.array(pred) / 255
    pred = pred.astype(np.uint8)
    if resize:
        target.resize((int(0.5 * target.size[0]), int(0.5 * target.size[1])))

    target = np.array(target, dtype=np.uint8)[:pred.shape[0], :pred.shape[1]]
    index = target == 53
    target[index] = 0
    pred[index] = 0

    data_loader = DataLoader(data_path=path_original_data,
                             metadata_path=path_meta_data)
    color_dict = data_loader.color_dict_binary
    target_dict = data_loader.get_target_dict()
    annotations_dict = data_loader.annotations_dict

    labels = ['02', 'Abassamento', 'Abbassamento', 'Area Punture insetti', 'Area aperta', 'Area vene', 'Buco',
              'Cicatrice',
              'Cicatrice aperta', 'Contaminazione', 'Crease', 'Difetto di lavorazione', 'Dirt', 'Fianco',
              'Fiore marcio',
              'Insect bite', 'Marchio', 'Microcut', 'Piega', 'Pinza', 'Pinze', 'Poro', "Puntura d'insetto",
              'Puntura insetto', 'Ruga', 'Rughe', 'Scopertura', 'Scratch', 'Smagliatura', 'Soffiatura', 'Struttura',
              'Taglio', 'Vena', 'Vene', 'Verruca', 'Wart', 'Zona aperta', 'verruca']

    metrics = [StreamSegMetrics(2), StreamSegMetrics(2), StreamSegMetrics(2)]
    false_positives = 0
    true_negatives = [0, 0]
    errors = np.array([[0, 0], [0, 0]])
    errors, false_positives, metric, target_color, pred_color, true_negatives = error_count(None,
                                                                                            pred,
                                                                                            target, data_loader,
                                                                                            labels, errors,
                                                                                            false_positives,
                                                                                            true_negatives,
                                                                                            metrics, resize=False,
                                                                                            size=None,
                                                                                            scale=None,
                                                                                            centercrop=False, path=path)
    exp = 'crop_all_classes'

    Image.fromarray(pred_color.astype(np.uint8)).save(
        save_path + r'/rh_{}_pred_color.png'.format(exp), format='PNG')
    Image.fromarray(target_color.astype(np.uint8)).save(
        save_path + r'/rh_{}_mask_color.png'.format(exp), format='PNG')

    labels = ['Insect bite', 'Binary', 'Good Area']
    new_list = [
        label + '\n' + '\n'.join([f"{name}, {performance}" for name, performance in metric[i].get_results().items()])
        for
        i, label in enumerate(labels)]
    string = '\n\n'.join(
        new_list) + f'\n\nBinary: {errors[0]} \nInsect Bite: {errors[1]}'
    f = open(os.path.join(save_path, 'rh_{}_performance'.format(exp)), 'w')
    f.write(string)



