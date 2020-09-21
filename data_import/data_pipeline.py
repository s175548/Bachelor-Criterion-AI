import os, numpy as np,cv2
from data_import.data_loader import DataLoader,get_patches


""" Steps:
1. Import complete dataset img + mask. Good and defect areas (You can choose to pick visibility score = X only)
2. Find background and add to mask
3. Crop images and masks to NxN size (default: 256x256)
4. Add bounding boxes to 3.
5. Divide into test and training
6. Random crop to N1xN1 (default: 200x200) and flip vertically and horizontally with probability 0.5 for both (independently) (+ whitening)
"""

def import_data_and_mask(data_loader,directory_path=r'C:\Users\Mads-\Documents\Universitet\5. Semester\Bachelorprojekt\data_folder',visibility_scores = []):
    if visibility_scores != []:
        idx = np.array([])
        for score in visibility_scores:
            idx = np.hstack((idx, np.where(np.array(dataloader.visibility_score) == score)[0]))
        idx = np.sort(idx)
    else:
        idx = data_loader.valid_annotations
        
    for i in idx:
        img,mask = data_loader.get_image_and_labels(i)

    #     os.chdir(directory_path)
    #     img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # cv2 has BGR channels, and Pillow has RGB channels, so they are transformed here
    #     im_pil = Image.fromarray(img2)
    #     im_pil.save(str(i)+".jpg")
    #     os.chdir(r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/training_mask')
    #     _, binary = cv2.threshold(mask * 255, 225, 255, cv2.THRESH_BINARY_INV)
    #     mask_pil = Image.fromarray(binary)
    #     mask_pil.convert('RGB').save(str(i)+'_mask.png')


"""TO DO: 
Extract good areas, that does not have any segmentations.
Fix border area in background mask (The border is now flawless)
"""
if __name__ == "__main__":
    dataloader = DataLoader()
    import_data_and_mask(dataloader,visibility_scores=[2,3])
    



