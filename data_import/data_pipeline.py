import os, numpy as np,cv2,random
from data_import.data_loader import DataLoader
from PIL import Image
from data_import.masks_to_bounding_box import get_background_mask,convert_mask_to_bounding_box
from semantic_segmentation.DeepLabV3.utils.ext_transforms import ExtEnhanceContrast

""" Steps:
1. Import complete dataset img + mask. Good and defect areas (You can choose to pick visibility score = X only)
2. Find background and add to mask
3. Crop images and masks to NxN size (default: 256x256)
4. Add bounding boxes to 3.
5. Divide into test and training
6. Random crop to N1xN1 (default: 200x200) and flip vertically and horizontally with probability 0.5 for both (independently) (+ whitening)
"""

def import_data_and_mask(data_loader,idx_to_consider='All',labels="All",path=None,visibility_scores = [2,3],exclude_no_mask_crops=True,make_binary=True,ignore_good=False,crop=False):
    if visibility_scores!= "All":
        visibility_idx=data_loader.get_visibility_score(visibility_scores)
        idx=visibility_idx
    if labels != "All":
        label_idx=data_loader.get_index_for_label(labels)
        idx=label_idx
    if (labels != "All") and (visibility_scores != "All"):
        idx=np.intersect1d(label_idx,visibility_idx)
    if idx_to_consider != 'All':
        idx = np.intersect1d(idx_to_consider, idx)


    for i in idx:
        i = int(i)
        img,mask = data_loader.get_image_and_labels([i],labels=labels,make_binary=make_binary,ignore_good=ignore_good)
        img_crops, mask_crops= data_loader.generate_patches(img[0],mask[0],img_index=i)
        if crop==True:
            for k in range(len(img_crops)):
                if exclude_no_mask_crops:
                    if list(np.setdiff1d(np.unique(mask_crops[k]),[0,121,  98,  62]))==[]:
                        pass
                    else:
                        k = int(k)
                        im_pil = Image.fromarray(img_crops[k])
                        im_pil.save( os.path.join(path,str(i)+"_"+str(k) + ".png") )
                        mask_pil = Image.fromarray(mask_crops[k])
                        mask_pil.save(os.path.join( path, str(i)+"_"+str(k) + '_mask.png'))
        else:
            im_pil = Image.fromarray(img[0])
            im_pil.save(os.path.join(path, str(i) +".png"))
            mask_pil = Image.fromarray(mask[0])
            mask_pil.save(os.path.join(path, str(i) + '_mask.png'))







"""TO DO:
Extract good areas, that does not have any segmentations.
Fix border area in background mask (The border is now flawless)
"""
if __name__ == "__main__":
     # data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
    #data_loader = DataLoader()
    data_loader = DataLoader(
         data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',
         metadata_path=r'samples/model_comparison.csv')

#import_data_and_mask(data_loader,path="/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/",visibility_scores=[2,3],labels=['Puntura insetto'])
    #import_data_and_mask(data_loader,path=r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi',labels=['Piega','Verruca','Puntura insetto','Background'],make_binary=False)
    import_data_and_mask(data_loader,path="/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/",make_binary=True)
     
 #   data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
     #import_data_and_mask(data_loader,path="/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/",visibility_scores=[1,2,3],labels=['Puntura insetto','Background'],make_binary=True)
     #import_data_and_mask(data_loader,path=r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi',labels=['Puntura insetto','Background'],make_binary=True)
#    import_data_and_mask(data_loader,path=r"C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_30_09",visibility_scores=[2,3],labels=['Puntura insetto'])
