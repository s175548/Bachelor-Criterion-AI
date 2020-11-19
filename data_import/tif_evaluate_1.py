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
Image.MAX_IMAGE_PIXELS = None
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#'/RED_HALF02_grain_01_v.tif'
#'/WALKNAPPA_VDA_04_grain_01_v.tif'


binary=True
device=torch.device('cuda')
model_resize=True
resize=False
model_name='DeepLab'
n_classes=1
patch_size=1024
overlap=512
Villads=False
HPC=True
step_size=2

def output_model(img_array):
    image = Image.fromarray(img_array)
    image, _ = transform_function(image, label)
    image = image.unsqueeze(0).to(device, dtype=torch.float32)
    return model(image)['out'].detach().cpu().squeeze().numpy()



if Villads:
    path_original_data = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
    path_train = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train"
    path_val = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/val"
    path_meta_data = r'samples/model_comparison.csv'
    save_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images'
    tif_path = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /tif_images'
    model_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /models/binær_several_classes/DeepLab_backbone_exp0.01.pt'

elif HPC:
    path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
    path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train' ###
    path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'     ###
    path_meta_data = r'samples/model_comparison.csv'
    save_path = r'/work3/s173934/Bachelorprojekt/tif_img'
    tif_path= r'/work3/s173934/Bachelorprojekt/tif_img'
    model_path=r'/work3/s173934/Bachelorprojekt/exp_results/resize_vs_randomcrop/all_class_dataset/resize/DeepLab_extended_dataset_resize_true0.01.pt'




#'/WALKNAPPA_VDA_04_grain_01_v.tif'
#'/RED_HALF02_grain_01_v.tif'

data_loader = DataLoader(data_path=path_original_data, metadata_path=path_meta_data)
image=load_tif_as_numpy_array(tif_path+'/RED_HALF02_grain_01_v.tif')
split_imgs, split_x_y,patch_dim = data_loader.generate_tif_patches(image, patch_size=patch_size,
                                                                         sliding_window=overlap)


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


if resize:
    transform_function = et.ExtCompose(
        [et.ExtResize(scale=0.5),
         et.ExtEnhanceContrast(),
         et.ExtToTensor(),
         et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

else:
    transform_function = et.ExtCompose([et.ExtEnhanceContrast(),
                                        et.ExtToTensor(),
                                        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if resize:
    patch_dim=(int(patch_dim[0]*0.5),int(patch_dim[1]*0.5))
    overlap=int(0.5*overlap)



target_tif=[]
label=Image.fromarray(np.zeros(patch_dim,dtype=np.uint8))
for i in range(0,split_x_y[0],step_size):
    print(i)
    pred_stack=[]
    for j in range(0,split_x_y[1],step_size):
        print(j)
        output = output_model(img_array=split_imgs[i * split_x_y[1] + j])
        l_slice,r_slice=(slice(0,None),slice(0,None),slice(0,overlap)),(slice(0,None),slice(0,None),slice(patch_dim[1]-overlap,patch_dim[1]))
        t_slice,b_slice=(slice(0,None),slice(patch_dim[0]-overlap,patch_dim[0]),slice(0,None)),(slice(0,None),slice(0,overlap),slice(0,None))
        if j != 0:
            output_l=output_model(img_array=split_imgs[i*split_x_y[1]+j-1])[r_slice]
            output[l_slice]=(output[l_slice]+output_l)/2
        if i != 0:
            output_t = output_model(img_array=split_imgs[(i-1) * split_x_y[1] + j])[b_slice]
            output[t_slice] = (output[t_slice]  + output_t) / 2
        if j != split_x_y[1]-1:
            output_r = output_model(img_array=split_imgs[i * split_x_y[1] + j+1])[l_slice]
            output[r_slice] = (output[r_slice] + output_r) / 2
        if i < split_x_y[0]-2:
            output_b = output_model(img_array=split_imgs[(i+1) * split_x_y[1] + j])[t_slice]
            output[b_slice] = (output[b_slice] + output_b) / 2
        pred = np.argmax(output,axis=0)

        if isinstance(pred_stack,list):
            pred_stack=pred
        else:
            pred_stack=np.hstack((pred_stack,pred))

    if isinstance(target_tif,list):
        target_tif=pred_stack
    else:
        target_tif=np.vstack((target_tif,pred_stack))

PIL.Image.fromarray(target_tif.astype(np.uint8)*255).save(tif_path+'/RF_AC_resize.png')

