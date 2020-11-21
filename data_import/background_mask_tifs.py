from data_import.data_loader import get_background_mask, DataLoader
import PIL
import numpy as np

img_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/RED_HALF02_grain_01_v.tif'
mask_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/RF_AC_no_resize_pred/RF_AC_no_resize_pred_color.png'
img_array=np.array(PIL.Image.open(img_path))
mask_array=np.array(PIL.Image.open(mask_path))
data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',
                             metadata_path=r'samples/model_comparison.csv')
back_mask=get_background_mask(img_array)
back_mask=back_mask[:mask_array.shape[0],:mask_array.shape[1]]
back_mask = np.dstack((back_mask, back_mask, back_mask))
mask_array[(np.array(back_mask)!=0) & (mask_array!=0)]=0
mask = mask_array + np.array(back_mask) / 255 * data_loader.annotations_dict["Background"]
mask_3d=mask
label='Background'
color_map_dict = data_loader.color_dict_binary
color_id = data_loader.annotations_dict[label]
color_map = color_map_dict[color_id]
index = mask[:,:,0] == color_id
mask_3d[index, :] = mask_3d[index, :] / color_id * color_map
mask_3d=PIL.Image.fromarray(mask_3d.astype(np.uint8)).resize((int(mask.shape[1]*0.1),int(mask.shape[0]*0.1)))
mask_3d.save('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/RF_AC_no_resize_pred/RF_AC_pred_pred.png')




