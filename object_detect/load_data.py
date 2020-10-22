import sys
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')

from data_import.data_loader import DataLoader
from data_import.data_pipeline import import_data_and_mask

if __name__ == "__main__":
    save_path_train = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\multi\train'
    save_path_val = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\multi\val'
    #binary_path = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary'
    #tick_path = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\tick_bite'

    data_loader = DataLoader(
         data_path=r'C:\Users\johan\OneDrive\Skrivebord\leather_patches',
         metadata_path=r'samples\model_comparison.csv')

    train, val = data_loader.test_training_split()

    #import_data_and_mask(data_loader, idx_to_consider=train,
    #                     path=save_path_train,
    #                     labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
    import_data_and_mask(data_loader, idx_to_consider=val,
                         path=save_path_val,
                         labels=['Piega', 'Verruca', 'Puntura insetto', 'Background'], make_binary=False)
    #import_data_and_mask(data_loader=data_loader,
    #                     path=binary_path,
    #                     visibility_scores=[2,3],
    #                     labels = ['Piega', 'Verruca', 'Puntura insetto', 'Background'],
    #                     make_binary = True)
    #import_data_and_mask(data_loader=data_loader,
    #                    path=tick_path,
    #                     visibility_scores=[2,3],
    #                     labels = ['Puntura insetto', 'Background'],
    #                     make_binary = True)


