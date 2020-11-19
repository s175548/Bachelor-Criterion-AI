import sys
sys.path.append('/zhome/db/f/128823/Bachelor/Bachelor-Criterion-AI')

from semantic_segmentation.DeepLabV3.performance_metric_function import error_count
import PIL
from data_import.data_loader import DataLoader
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics
villads=False
HPC=True
resize=False

if villads:
    path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/annotations_RED_HALF02_grain_01_v.tif.json'
    pred=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/red_half_02_01_all_classes.png')
    target=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/RED_HALF02_grain_01_v_target_1d.png')
    save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images/'
    path_meta_data=r'samples/model_comparison.csv'
    path_original_data=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
elif HPC:
    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches' ###
    path_meta_data = r'samples/model_comparison.csv'
    save_path = r'/work3/s173934/Bachelorprojekt/tif_img'
    path = r'/work3/s173934/Bachelorprojekt/tif_img/annotations_RED_HALF02_grain_01_v.tif.json'
    pred=PIL.Image.open('/work3/s173934/Bachelorprojekt/tif_img/red_half_02_01_all_classes_sliding_window.png')
    target=PIL.Image.open('/work3/s173934/Bachelorprojekt/tif_img/RED_HALF02_grain_01_v_target_1d.png')

pred=np.array(pred)/255
pred=pred.astype(np.uint8)

if resize:
    target.resize((int(0.5*pred.shape[1]),int(0.5*pred.shape[1])))

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
                                                                                        metrics, resize=False, size=None,
                                                                                        scale=None, centercrop=False,path=path)



PIL.Image.fromarray(pred_color.astype(np.uint8)).save(
    save_path + r'/RF_AC_no_resize_pred_color.png', format='PNG')
PIL.Image.fromarray(target_color.astype(np.uint8)).save(
    save_path + r'/RF_AC_no_resize_mask_color.png', format='PNG')

labels = ['Insect bite', 'Binary', 'Good Area']
new_list = [
    label + '\n' + '\n'.join([f"{name}, {performance}" for name, performance in metric[i].get_results().items()]) for
    i, label in enumerate(labels)]
string = '\n\n'.join(
    new_list) + f'\n\nBinary: {errors[0]} \nInsect Bite: {errors[1]}'
f = open(os.path.join(save_path, 'performance'), 'w')
f.write(string)