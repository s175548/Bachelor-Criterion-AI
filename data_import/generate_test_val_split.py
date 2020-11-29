import sys, os
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')
from data_import.data_pipeline import import_data_and_mask
from data_import.data_loader import DataLoader
import os

Villads = True
Johannes = False
if Villads:
    data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
    save_path_train = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train_good"
    save_path_val = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/val_new"

elif Johannes:
    path_original_data = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches'
    path_meta_data = r'samples/model_comparison.csv'
    data_loader = DataLoader(data_path=path_original_data,
                                 metadata_path=path_meta_data)
    save_path_train = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\full_scale\train'
    save_path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\full_scale\val'
else:
    data_loader = DataLoader()
    save_path_train =r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\test_data_binary_vis_2_and_3\train'
    save_path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\test_data_binary_vis_2_and_3\val'

train,val=data_loader.test_training_split(villads_dataset=True)

"""if you want to split by the skin"""
train,val=data_loader.test_training_split_skin()

labels=['Good','Good Area','Good Area_grain01','Good Area_grain04','Good Area_grain05',
 'Good Area_grain06',
 'Good Area_grain07',
 'Good Area_grain09',
 'Good area',
 'Good area_grain08',
 'Good area_grain10',]

import_data_and_mask(data_loader,idx_to_consider=val,
                    path=save_path_val, make_binary=True,exclude_no_mask_crops=False,crop=False)

#import_data_and_mask(data_loader,idx_to_consider=train,
#                     path=save_path_train, make_binary=True,exclude_no_mask_crops=True,labels=labels,visibility_scores=[2,3],crop=True)
