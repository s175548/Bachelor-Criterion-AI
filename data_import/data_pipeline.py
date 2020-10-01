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

def import_data_and_mask(data_loader,labels="All",path=None,visibility_scores = "All",exclude_no_mask_crops=True,make_binary=True):
    if visibility_scores != "All":
        visibility_idx=data_loader.get_visibility_score()
        idx=visibility_idx
    if labels != "All":
        label_idx=data_loader.get_index_for_label(labels)
        idx=label_idx
    if (labels != "All") and (visibility_scores != "All"):
        idx=np.intersect1d(label_idx,visibility_idx)

    for i in idx:
        i = int(i)
        img,mask = data_loader.get_image_and_labels([i],labels=labels,make_binary=make_binary)
        img_crops, mask_crops= data_loader.generate_patches(img[0],mask[0],img_index=i)

        for k in range(len(img_crops)):
            if exclude_no_mask_crops:
                if list(np.setdiff1d(np.unique(mask_crops[k]),[0,53, 101, 113]))==[]:
                    pass
                else:
                    k = int(k)
                    im_pil = Image.fromarray(img_crops[k])
                    im_pil.save( os.path.join(path,str(i)+"_"+str(k) + ".png") )
                    mask_pil = Image.fromarray(mask_crops[k])
                    mask_pil.save( os.path.join( path, str(i)+"_"+str(k) + '_mask.png') )

        # bounding_boxes = [convert_mask_to_bounding_box(mask_crops[i]) for i in range(len(mask_crops))]

        #Add whitening, random crop, flip





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
     data_loader = DataLoader(
         data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',
         metadata_path=r'samples/model_comparison.csv')
#    data_loader = DataLoader()
   # import_data_and_mask(data_loader,path="/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/",visibility_scores=[2,3],labels=['Puntura insetto'])
     import_data_and_mask(data_loader,path="/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/",visibility_scores=[2,3],labels=['Puntura insetto'],make_binary=True)
#    import_data_and_mask(data_loader,path=r"C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_30_09",visibility_scores=[2,3],labels=['Puntura insetto'])
