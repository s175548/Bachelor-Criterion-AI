import sys
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')


from object_detect.leather_data_hpc import LeatherData
from torch.utils import data
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
import os, torch
import numpy as np
import random


transform_function = et.ExtCompose([et.ExtToTensor()])


if __name__ == "__main__":
    #path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_28_09/mask'
    #path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_28_09/img'

    batch_size = 128
    #val_batch_size = 64
    #C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\From_HPC
    path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\mask'
    path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\img'

    batch_size = 4
    val_batch_size = 4
    num_epoch = 1

    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[:-4] != ".DS_S"])
    N_files = len(file_names)
    shuffled_index = np.random.permutation(len(file_names))
    file_names_img = file_names[shuffled_index]
    file_names = file_names[file_names != ".DS_S"]

    # Define dataloaders
    train_dst = LeatherData(path_mask=path_mask, path_img=path_img,
                            list_of_filenames=file_names[:round(N_files * 0.10)], transform=transform_function)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img,
                          list_of_filenames=file_names[round(N_files * 0.80):], transform=transform_function)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)
