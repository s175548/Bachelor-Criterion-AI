import sys
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')

import os
from data_import.data_loader import DataLoader
from data_import.data_pipeline import import_data_and_mask
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
import torch
import object_detect.helper.utils as utils
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
from torch.utils import data
import random
import numpy as np

transform_function = et.ExtCompose([et.ExtRandomCrop(scale=0.7),et.ExtRandomHorizontalFlip(p=0.5),et.ExtRandomVerticalFlip(p=0.5),et.ExtEnhanceContrast(),et.ExtToTensor()])

random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cpu')
lr = 0.01
layers_to_train = 'classifier'
num_epoch = 1
path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
path_meta_data = r'samples/model_comparison.csv'
optim = "SGD"
path_train = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\full_scale\train'
path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\full_scale\val'
dataset = "binary_scale"

path_save = '/Users/johan/iCloudDrive/DTU/KID/BA/Kode/FRCNN/'
save_folder = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Predictions_FRCNN'

print("Device: %s" % device)
data_loader = DataLoader(data_path=path_original_data,
                         metadata_path=path_meta_data)

color_dict = data_loader.color_dict_binary
target_dict = data_loader.get_target_dict()
annotations_dict = data_loader.annotations_dict

batch_size = 10
val_batch_size = 2

file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] != "k"])
N_files = len(file_names_train)
#shuffled_index = np.random.permutation(len(file_names_train))
#file_names_train = file_names_train[shuffled_index]
file_names_train = file_names_train[file_names_train != ".DS_S"]

file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
N_files = len(file_names_val)

train_dst = LeatherData(path_mask=path_train, path_img=path_train, list_of_filenames=file_names_train,
                        bbox=True,
                        transform=transform_function, color_dict=color_dict, target_dict=target_dict)
val_dst = LeatherData(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                      bbox=True,
                      transform=transform_function, color_dict=color_dict, target_dict=target_dict)

train_loader = data.DataLoader(
    train_dst, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
val_loader = data.DataLoader(
    val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))


if __name__ == "__main__":

    print("HE")