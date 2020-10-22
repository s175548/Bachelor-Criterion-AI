import sys, os
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')
from data_import.data_pipeline import import_data_and_mask
from data_import.data_loader import DataLoader

Villads = False
Johannes = True
if Villads:
    data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
    save_path_train = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train"
    save_path_val = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/val"

elif Johannes:
    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
    path_meta_data = r'samples/model_comparison.csv'
    data_loader = DataLoader(data_path=path_original_data,
                                 metadata_path=path_meta_data)
    save_path_train = r'/zhome/dd/4/128822/Bachelorprojekt/multi/train'
    save_path_val = r'/zhome/dd/4/128822/Bachelorprojekt/multi/val'
else:
    data_loader = DataLoader()
    save_path_train =r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_binary_vis_2_and_3\train'
    save_path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_binary_vis_2_and_3\val'
train,val=data_loader.test_training_split()


import_data_and_mask(data_loader,idx_to_consider=train,
                     path=save_path_train,
                     labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
import_data_and_mask(data_loader,idx_to_consider=val,
                     path=save_path_val,
                     labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
