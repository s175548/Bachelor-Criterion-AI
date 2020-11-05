import sys
#sys.path.append('/zhome/87/9/127623/BachelorProject/cropped_data/Bachelor-Criterion-AI')
#sys.path.append('/zhome/87/9/127623/BachelorProject/cropped_data/Bachelor-Criterion-AI/semantic_segmentation')

sys.path.append('/zhome/db/f/128823/Bachelor/Bachelor-Criterion-AI')

from data_import.data_loader import DataLoader
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import PIL
from semantic_segmentation.DeepLabV3.network.modeling import _segm_mobilenet
import torchvision.transforms.functional as F
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics

from data_import.data_loader import convert_to_image


def error_count(idx, pred_color, target_color, data_loader, labels, errors, false_positives, metric, reize=False,
                size=None, scale=None):
    pred = pred_color.copy()
    target = target_color.copy()
    if np.sum(target == 1) != 0:
        masks = data_loader.get_separate_segmentations(
            os.path.join(data_loader.data_path, data_loader.metadata_csv[idx, 3][1:]), labels=labels)
        buffer = 42
        xdim_s = []
        ydim_s = []
        for mask in masks:
            label, mask = mask[0], np.squeeze(np.array(mask[1]).astype(np.uint8))
            mask = F.center_crop(PIL.Image.fromarray(mask), output_size=size)
            mask = np.array(mask)
            if reize:
                resize_shape = (int(mask.shape[0] * scale), int(mask.shape[1] * scale) * scale)
                mask = F.resize(PIL.Image.fromarray(mask), resize_shape, PIL.Image.NEAREST)
            mask = np.array(mask)
            row, col = np.where(mask != 0)
            xdim = (np.maximum(np.min(row) - buffer, 0), np.minimum(np.max(row) + buffer, mask.shape[0]))
            xdim_s.append(xdim)
            ydim = (np.maximum(np.min(col) - buffer, 0), np.minimum(np.max(col) + buffer, mask.shape[1]))
            ydim_s.append(ydim)
            mask = mask[xdim[0]:xdim[1], ydim[0]:ydim[1]]
            defect_found = int(np.sum(pred[xdim[0]:xdim[1], ydim[0]:ydim[1]] != 0) > 0)
            if label == 'Insect bite':
                errors[1, 0] += 1
                errors[1, 1] += defect_found
                metric[0].update(mask, pred[xdim[0]:xdim[1], ydim[0]:ydim[1]])
            else:
                errors[0, 0] += 1
                errors[0, 1] += defect_found
                metric[1].update(mask, pred[xdim[0]:xdim[1], ydim[0]:ydim[1]])
        for xdim, ydim in zip(xdim_s, ydim_s):
            pred[xdim[0]:xdim[1], ydim[0]:ydim[1]] = 0
            target[xdim[0]:xdim[1], ydim[0]:ydim[1]] = 0
    else:
        xdim_s = None
        ydim_s = None
    if np.sum(pred == 1) > 0:
        false_positives += 1
    target[target == 2] = 0
    metric[2].update(target, pred)
    target_color[target_color == 2] = 0
    target_color, pred_color = color_target_pred(target_color, pred_color, pred, xdim_s, ydim_s)
    return errors, false_positives, metric, target_color, pred_color


def color_target_pred(target, pred, pred_false_pos, xdim_s, ydim_s):
    target_tp = np.zeros(target.shape)
    target_fp = np.zeros(target.shape)
    fill = 3
    if xdim_s != None:
        for xdim, ydim in zip(xdim_s, ydim_s):
            pred_crop = pred[xdim[0]:xdim[1], ydim[0]:ydim[1]]
            pred[xdim[0]:xdim[1], ydim[0]:ydim[1]][pred_crop == 1] = 255
            if np.sum(pred[xdim[0]:xdim[1], ydim[0]:ydim[1]] != 0) > 0:
                target_tp[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 255
                target[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 0
                target_tp[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 255
                target[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 0
                target_tp[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 255
                target[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 0
                target_tp[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 255
                target[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 0
            else:
                target_fp[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 255
                target[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 0
                target_fp[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 255
                target[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 0
                target_fp[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 255
                target[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 0
                target_fp[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 255
                target[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 0

    pred_false_pos[pred_false_pos == 1] = 255
    pred_rgb = np.dstack((pred_false_pos, pred, np.zeros(pred.shape)))
    target_rgb = np.dstack((target_fp, target_tp, target * 255))
    return target_rgb, pred_rgb


"""Arguments"""

Villads = False
HPC = True
model_name = 'DeepLab'
n_classes = 1
resize = False
size = 1024
scale = 0.5
binary = True
device = torch.device('cuda')
data_set = 'val'

if Villads:
    path_original_data = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
    path_train = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train"
    path_val = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/val"
    path_meta_data = r'samples/model_comparison.csv'
    save_path = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /model_predictions'
    model_path = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /models/binær_several_classes/DeepLab_backbone_exp0.01.pt'
elif HPC:
    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
    path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train' ###
    path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'     ###
    path_meta_data = r'samples/model_comparison.csv'
    save_path = r'/zhome/db/f/128823/Bachelor/data_all_classes/resized_model'          ###
    model_path = r'/work3/s173934/Bachelorprojekt/exp_results/original_res/DeepLab_res_exp0.01.pt'       ###
else:
    path_original_data = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\leather_patches'
    path_train = r"C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\tif_images"
    path_val = r"C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\tif_images"
    path_meta_data = r'samples/model_comparison.csv'
    save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\slet\predictions'
    model_path = r'E:\downloads_hpc_bachelor\exp_results\backbone\classifier_only\ResNet\DeepLab_backbone_exp0.01.pt'

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

if model_name == 'DeepLab':
    model = deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
    model.classifier[-1] = torch.nn.Conv2d(256, n_classes + 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
    model.aux_classifier[-1] = torch.nn.Conv2d(256, n_classes + 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
else:
    model = _segm_mobilenet('deeplabv3', 'mobile_net', output_stride=8, num_classes=n_classes + 2,
                            pretrained_backbone=True)

model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

data_loader = DataLoader(data_path=path_original_data, metadata_path=path_meta_data)

file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] != "k"])
file_names_train = file_names_train[file_names_train != ".DS_S"]

file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
file_names_val = file_names_val[file_names_val != ".DS_S"]

transform_function = et.ExtCompose([et.ExtCenterCrop(size=size),
                                    et.ExtEnhanceContrast(),
                                    et.ExtToTensor(),
                                    et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_function_resize = et.ExtCompose([et.ExtCenterCrop(size=size),
                                           et.ExtResize(scale=scale),
                                           et.ExtEnhanceContrast(),
                                           et.ExtToTensor(),
                                           et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
if binary:
    color_dict = data_loader.color_dict_binary
    target_dict = data_loader.get_target_dict()
    annotations_dict = data_loader.annotations_dict

else:
    color_dict = data_loader.color_dict
    target_dict = data_loader.get_target_dict(labels)
    annotations_dict = data_loader.annotations_dict

train_dst = LeatherData(path_mask=path_train, path_img=path_train, list_of_filenames=file_names_train,
                        transform=transform_function, color_dict=color_dict, target_dict=target_dict)
val_dst = LeatherData(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
                      transform=transform_function, color_dict=color_dict, target_dict=target_dict)

train_images = []

if data_set == 'train':
    for i in range(len(train_dst)):
        train_images.append(train_dst.__getitem__(i))

elif data_set == 'val':
    for i in range(len(val_dst)):
        train_images.append(val_dst.__getitem__(i))
        if i == 15:
            break

labels = ['02', 'Abassamento', 'Abbassamento', 'Area Punture insetti', 'Area aperta', 'Area vene', 'Buco', 'Cicatrice',
          'Cicatrice aperta', 'Contaminazione', 'Crease', 'Difetto di lavorazione', 'Dirt', 'Fianco', 'Fiore marcio',
          'Insect bite', 'Marchio', 'Microcut', 'Piega', 'Pinza', 'Pinze', 'Poro', "Puntura d'insetto",
          'Puntura insetto', 'Ruga', 'Rughe', 'Scopertura', 'Scratch', 'Smagliatura', 'Soffiatura', 'Struttura',
          'Taglio', 'Vena', 'Vene', 'Verruca', 'Wart', 'Zona aperta', 'verruca']

metrics = [StreamSegMetrics(2), StreamSegMetrics(2), StreamSegMetrics(2)]
false_positives = 0
errors = np.array([[0, 0], [0, 0]])

for i in range(len(train_images)):
    print(i)
    image = train_images[i][0].unsqueeze(0)
    target = train_images[i][1]
    image = image.to(device, dtype=torch.float32)
    if model_name == 'DeepLab':
        output = model(image)['out']
    else:
        output = model(image)
    pred = output.detach().max(dim=1)[1].cpu().squeeze().numpy()
    target = target.squeeze().numpy()

    errors, false_positives, metric, target_color, pred_color = error_count(int(file_names_val[i]), pred.copy(),
                                                                            target.copy(), data_loader, labels, errors,
                                                                            false_positives, metrics, resize, size=size,
                                                                            scale=scale)

    target = convert_to_image(target.squeeze(), color_dict, target_dict)
    pred = convert_to_image(pred.squeeze(), color_dict, target_dict)
    image = (denorm(train_images[i][0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
    PIL.Image.fromarray(image.astype(np.uint8)).save(os.path.join(save_path + r'/{}_img.png'.format(file_names_val[i])),format='PNG')
    PIL.Image.fromarray(pred_color.astype(np.uint8)).save(os.path.join(save_path ,  r'/{}_pred_color.png'.format(file_names_val[i])),format='PNG')
    PIL.Image.fromarray(target_color.astype(np.uint8)).save(os.path.join(save_path , r'/{}_mask_color.png'.format(file_names_val[i])),format='PNG')
    # PIL.Image.fromarray(image.astype(np.uint8)).save(
    #     os.path.join(save_path, r'binary', model_name, data_set + '1', r'{}_img.png'.format(file_names_val[i])),
    #     format='PNG')
    # PIL.Image.fromarray(pred_color.astype(np.uint8)).save(
    #     os.path.join(save_path, r'binary', model_name, data_set + '1', r'{}_pred_color.png'.format(file_names_val[i])),
    #     format='PNG')
    # PIL.Image.fromarray(target_color.astype(np.uint8)).save(
    #     os.path.join(save_path, r'binary', model_name, data_set + '1', r'{}_mask_color.png'.format(file_names_val[i])),
    #     format='PNG')
#    PIL.Image.fromarray(pred.astype(np.uint8)).save(
#        os.path.join(save_path, r'binary', model_name, data_set + '1', r'{}_pred.png'.format(file_names_val[i])),
#        format='PNG')
#    PIL.Image.fromarray(target.astype(np.uint8)).save(
#        os.path.join(save_path, r'binary', model_name, data_set + '1', r'{}_mask.png'.format(file_names_val[i])),
#        format='PNG')

labels = ['Insect bite', 'Binary', 'Good Area']
new_list = [
    label + '\n' + '\n'.join([f"{name}, {performance}" for name, performance in metric[i].get_results().items()]) for
    i, label in enumerate(labels)]
string = '\n\n'.join(
    new_list) + f'\n\nBinary: {errors[0]} \nInsect Bite: {errors[1]} \nFalse positives: {false_positives}'
f = open(os.path.join(save_path, 'performance'), 'w')
f.write(string)
