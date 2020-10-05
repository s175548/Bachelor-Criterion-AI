import sys
sys.path.append('/zhome/dd/4/128822/Bachelorprojekt/Bachelor-Criterion-AI')

from data_import.data_loader import DataLoader
from data_import.data_pipeline import import_data_and_mask

if __name__ == "__main__":
    multi_path = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\multi'
    binary_path = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\binary'
    tick_path = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\cropped_data\tick_bite'
    data_loader = DataLoader(
         data_path=r'C:\Users\johan\OneDrive\Skrivebord\leather_patches',
         metadata_path=r'samples\model_comparison.csv')
    import_data_and_mask(data_loader=data_loader,
                         path= multi_path,
                         visibility_scores=[2,3], labels=['Piega','Verruca','Puntura insetto','Background'],
                         make_binary=False)
    import_data_and_mask(data_loader=data_loader,
                         path=binary_path,
                         visibility_scores=[2,3],
                         labels = ['Piega', 'Verruca', 'Puntura insetto', 'Background'],
                         make_binary = True)
    import_data_and_mask(data_loader=data_loader,
                         path=tick_path,
                         visibility_scores=[2,3],
                         labels = ['Puntura insetto', 'Background'],
                         make_binary = True)


