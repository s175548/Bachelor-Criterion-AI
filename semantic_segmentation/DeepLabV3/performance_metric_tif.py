import sys
sys.path.append('/zhome/db/f/128823/Bachelor/Bachelor-Criterion-AI')

from semantic_segmentation.DeepLabV3.performance_metric_function import error_count
import PIL
from data_import.data_loader import DataLoader, get_background_mask
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics

villads=False
HPC=True
resize=False


if villads:
    path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/annotations_RED_HALF02_grain_01_v.tif.json'
    pred=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/RF_3C_orig_res.png')
    target=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/RED_HALF02_grain_01_v_target_1d.png')
    save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/'
    path_meta_data=r'samples/model_comparison.csv'
    path_original_data=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
elif HPC:
    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches' ###
    path_meta_data = r'samples/model_comparison.csv'
    save_path = r'/work3/s173934/Bachelorprojekt/tif_img/RF_3C_no_resize_pred'
    path = r'/work3/s173934/Bachelorprojekt/tif_img/annotations_RED_HALF02_grain_01_v.tif.json'
    pred=PIL.Image.open('/work3/s173934/Bachelorprojekt/tif_img/RF_3C_no_resize_new.png')
    target=PIL.Image.open('/work3/s173934/Bachelorprojekt/tif_img/RED_HALF02_grain_01_v_target_1d.png')
    save_name='/RF_3C_no_resize'

if resize:
    target=target.resize((int(0.5*target.size[0]),int(0.5*target.size[1])))


pred=np.array(pred)/255
pred=pred.astype(np.uint8)
target=np.array(target,dtype=np.uint8)[:pred.shape[0],:pred.shape[1]]
index=target==53
target[index]=0
pred[index]=0

data_loader = DataLoader(data_path=path_original_data,
                             metadata_path=path_meta_data)
color_dict = data_loader.color_dict_binary
target_dict = data_loader.get_target_dict()
annotations_dict = data_loader.annotations_dict

labels = ['02', 'Abassamento', 'Abbassamento', 'Area Punture insetti', 'Area aperta', 'Area vene', 'Buco', 'Cicatrice',
          'Cicatrice aperta', 'Contaminazione', 'Crease', 'Difetto di lavorazione', 'Dirt', 'Fianco', 'Fiore marcio',
          'Insect bite', 'Marchio', 'Microcut', 'Piega', 'Pinza', 'Pinze', 'Poro', "Puntura d'insetto",
          'Puntura insetto', 'Ruga', 'Rughe', 'Scopertura', 'Scratch', 'Smagliatura', 'Soffiatura', 'Struttura',
          'Taglio', 'Vena', 'Vene', 'Verruca', 'Wart', 'Zona aperta', 'verruca']


metrics = [StreamSegMetrics(2), StreamSegMetrics(2), StreamSegMetrics(2)]
false_positives = 0
true_negatives = [0,0]
errors = np.array([[0, 0], [0, 0]])
errors, false_positives, metric, target_color, pred_color, true_negatives = error_count(None,
                                                                                        pred,
                                                                                        target, data_loader,
                                                                                        labels, errors,
                                                                                        false_positives, true_negatives,
                                                                                        metrics, resize=resize, size=None,
                                                                                        scale=0.5, centercrop=False,path=path)



PIL.Image.fromarray(pred_color.astype(np.uint8)).save(
    save_path + save_name + r'_pred_color.png', format='PNG')
PIL.Image.fromarray(target_color.astype(np.uint8)).save(
    save_path + save_name+ r'_mask_color.png', format='PNG')

labels = ['Insect bite', 'Binary', 'Good Area']
new_list = [
    label + '\n' + '\n'.join([f"{name}, {performance}" for name, performance in metric[i].get_results().items()]) for
    i, label in enumerate(labels)]
string = '\n\n'.join(
    new_list) + f'\n\nBinary: {errors[0]} \nInsect Bite: {errors[1]} \nFalse positives: {false_positives}'
f = open(os.path.join(save_path, 'performance_VDA_3C_no_resize'), 'w')
f.write(string)

img_list=[target_color.astype(np.uint8),pred_color.astype(np.uint8)]
img_name_list=[r'_mask_color_resized.png',r'_pred_color_resized.png']

for i,mask in enumerate(img_list):
    mask_3d=mask
    label='Background'
    color_map_dict = data_loader.color_dict_binary
    color_id = data_loader.annotations_dict[label]
    color_map = color_map_dict[color_id]
    mask_3d[index, :] = (mask_3d[index, :]+1) * color_map
    mask_3d=PIL.Image.fromarray(mask_3d.astype(np.uint8)).resize((int(mask.shape[1]*0.1),int(mask.shape[0]*0.1)))
    mask_3d.save(save_path + save_name+img_name_list[i],format='PNG')

