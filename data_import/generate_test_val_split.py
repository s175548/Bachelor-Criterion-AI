from data_import.data_pipeline import  import_data_and_mask
from data_import.data_loader import DataLoader


data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',
                             metadata_path=r'samples/model_comparison.csv')
train,val=data_loader.test_training_split()
import_data_and_mask(data_loader,idx_to_consider=train,
                     path="/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train",
                     labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
import_data_and_mask(data_loader,idx_to_consider=val,
                     path="/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/val",
                     labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
