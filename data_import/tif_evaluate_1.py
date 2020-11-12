import sys
sys.path.append('/zhome/db/f/128823/Bachelor/Bachelor-Criterion-AI')

from PIL import Image
from data_import.tif_import import load_tif_as_numpy_array
from data_import.data_loader import DataLoader
import numpy as np
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
import PIL
from semantic_segmentation.DeepLabV3.network.modeling import _segm_mobilenet
import os
import torchvision.transforms.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


binary=True
device=torch.device('cpu')
model_resize=True
resize=True
model_name='DeepLab'
n_classes=1
patch_size=128
Villads=True
HPC=False

  # Set padding to make better image predictions



if Villads:
    path_original_data = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
    path_train = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train"
    path_val = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/val"
    path_meta_data = r'samples/model_comparison.csv'
    save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /model_predictions'
    tif_path = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/img'
    model_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /models/binær_several_classes/DeepLab_backbone_exp0.01.pt'

elif HPC:
    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
    path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train' ###
    path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'     ###
    path_meta_data = r'samples/model_comparison.csv'
    save_path = r'/zhome/db/f/128823/Bachelor/data_all_classes/resized_model'
    tif_path='/zhome/db/f/128823/Bachelor/data_all_classes/tif_image'
    if model_resize:###
        model_path = r'/work3/s173934/Bachelorprojekt/exp_results/original_res/DeepLab_res_exp0.01.pt'
    else:
        model_path=r"work3/s173934/Bachelorprojekt/exp_results/binary_vs_multi/binary/ResNet/DeepLab_binary_exp0.01.pt"

data_loader = DataLoader(data_path=path_original_data, metadata_path=path_meta_data)
array = load_tif_as_numpy_array(tif_path+'/41.png')
split_imgs, split_x_y,patch_dimensions = data_loader.generate_tif_patches(array, patch_size=patch_size,
                                                                         padding=50,with_pad=True)

checkpoint=torch.load(model_path,map_location=device)

if model_name=='DeepLab':
    model=deeplabv3_resnet101(pretrained=True, progress=True,num_classes=21, aux_loss=None)
    model.classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
    model.aux_classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
else:
    model=_segm_mobilenet('deeplabv3', 'mobile_net', output_stride=8, num_classes=n_classes+2,pretrained_backbone=True)

model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()


if not resize:
    transform_function = et.ExtCompose([
                                        et.ExtEnhanceContrast(),
                                        et.ExtToTensor(),
                                        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
else:
    transform_function = et.ExtCompose(
        [et.ExtResize(scale=0.5),
         et.ExtEnhanceContrast(),
         et.ExtToTensor(),
         et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


target_tif=[]
for i in range(split_x_y[0]):
    print(i)
    pred_stack=[]
    for j in range(split_x_y[1]):
        print(j)
        label=Image.fromarray(np.zeros(split_imgs[i*split_x_y[1]+j].shape,dtype=np.uint8))
        image=Image.fromarray(split_imgs[i*split_x_y[1]+j].astype(np.uint8))
        if j==0:
            F.pad(image,padding=(0,0,50,0),padding_mode='reflect')
        if j==split_x_y[1]-1:
            F.pad(image, padding=(50, 0, 0, 0),padding_mode='reflect')
        if i==0:
            F.pad(image, padding=(0, 50, 0, 0), padding_mode='reflect')
        if i==split_x_y[0]-1:
            F.pad(image, padding=(0, 0, 0, 50), padding_mode='reflect')

        image,_=transform_function(image,label)
        image = image.unsqueeze(0).to(device, dtype=torch.float32)
        if model_name == 'DeepLab':
            output = model(image)['out']
        else:
            output = model(image)
        pred = output.detach().max(dim=1)[1].cpu().squeeze().numpy()


        pred=pred[50:-50,50:-50]
        if isinstance(pred_stack,list):
            pred_stack=pred
        else:
            pred_stack=np.hstack((pred_stack,pred))

    if isinstance(target_tif,list):
        target_tif=pred_stack
    else:
        target_tif=np.vstack((target_tif,pred_stack))

PIL.Image.fromarray(target_tif.astype(np.uint8)*255).save(tif_path+'/pred_521.png')

