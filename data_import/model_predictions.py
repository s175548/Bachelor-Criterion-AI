from data_import.data_loader import DataLoader
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
import PIL
from data_import.data_loader import convert_to_image
from semantic_segmentation.DeepLabV3.network.modeling import _segm_mobilenet
from torch.utils import data
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

batch_size= 16 # 16
val_batch_size= 4 #4

Villads=True
if Villads:
    path_original_data = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
    path_train = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data/train"
    path_val = r"/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /val"
    path_meta_data = r'samples/model_comparison.csv'
    save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /model_predictions'
    model_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /models/binær_several_classes/DeepLab_backbone_exp0.01.pt'
else:
    path_original_data = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\leather_patches'
    path_train = r"C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_all_classes\train"
    path_val = r"C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_all_classes\val"
    path_meta_data = r'samples/model_comparison.csv'
    save_path=r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\slet\predictions'
    model_path=r'E:\downloads_hpc_bachelor\exp_results\backbone\classifier_only\ResNet\DeepLab_backbone_exp0.01.pt'

checkpoint=torch.load(model_path,map_location=torch.device('cpu'))
model_name='DeepLab'
n_classes=1

if model_name=='DeepLab':
    model=deeplabv3_resnet101(pretrained=True, progress=True,num_classes=21, aux_loss=None)
    model.classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
    model.aux_classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
else:
    model=_segm_mobilenet('deeplabv3', 'mobile_net', output_stride=8, num_classes=n_classes+2,pretrained_backbone=True)


model.load_state_dict(checkpoint['model_state'])
model.eval()

data_loader = DataLoader(data_path=path_original_data ,metadata_path=path_meta_data)
labels =['Piega', 'Verruca', 'Puntura insetto' ,'Background']
binary=True
device=torch.device('cpu')

file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] != "k"])
file_names_train = file_names_train[file_names_train != ".DS_S"]

file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
file_names_val = file_names_val[file_names_val != ".DS_S"]

#transform_function = et.ExtCompose([et.ExtEnhanceContrast(), et.ExtRandomCrop((1000, 1000)), et.ExtToTensor(),
#                                   et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


transform_function = et.ExtCompose([
                                           et.ExtResize(scale=0.5),
                                           et.ExtRandomCrop(size=512,semantic_evaluation_resize=True,scale=0.7),
                                           et.ExtEnhanceContrast()])


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

train_loader = data.DataLoader(
    train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = data.DataLoader(
    val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)

train_images = []

data_set='val'
if data_set=='train':
    for i in range(len(train_dst)):
        train_images.append(train_dst.__getitem__(i))
elif data_set=='val':
    for i in range(len(val_dst)):
        train_images.append(val_dst.__getitem__(i))

data_set='val_all_class_resize'
for i in range(len(train_images)):
    print(i)
    image = train_images[i][0]
    image.save(os.path.join(save_path,r'multi',model_name,data_set,r'{}_img.png'.format(file_names_val[i])),format='PNG' )
    target = np.array(train_images[i][1])
    target = convert_to_image(target, color_dict, target_dict)
    PIL.Image.fromarray(target.astype(np.uint8)).save( os.path.join(save_path,r'multi',model_name,data_set,r'{}_mask.png'.format(file_names_val[i])),format='PNG' )




for i in range(len(train_images)):
    break
    image = train_images[i][0].unsqueeze(0)
    image = image.to(device, dtype=torch.float32)

    if model_name == 'DeepLab':
        output = model(image.float())['out']
    else:
        output = model(image)

    pred = output.detach().max(dim=1)[1].cpu().squeeze().numpy()
    target = train_images[i][1].cpu().squeeze().numpy()
    target = convert_to_image(target.squeeze(), color_dict, target_dict)
    pred = convert_to_image(pred.squeeze(), color_dict, target_dict)
    image = (denorm(train_images[i][0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
    PIL.Image.fromarray(image.astype(np.uint8)).save(os.path.join(save_path,r'multi',model_name,data_set,r'{}_img.png'.format(i)),format='PNG' )
    PIL.Image.fromarray(pred.astype(np.uint8)).save( os.path.join(save_path,r'multi',model_name,data_set,r'{}_pred.png'.format(i)),format='PNG' )
    PIL.Image.fromarray(target.astype(np.uint8)).save( os.path.join(save_path,r'multi',model_name,data_set,r'{}_mask.png'.format(i)),format='PNG' )
