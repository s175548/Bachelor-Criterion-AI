from data_import.data_pipeline import import_data_and_mask
from data_import.data_loader import DataLoader

Villads = False
if Villads:
    data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
    save_path_train = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train"
    save_path_val = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/val"
else:
    data_loader = DataLoader()
    save_path_train =r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi\train'
    save_path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi\val'
train,val=data_loader.test_training_split()


import_data_and_mask(data_loader,idx_to_consider=train,
                     path=save_path_train,
                     labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
import_data_and_mask(data_loader,idx_to_consider=val,
                     path=save_path_val,
                     labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
