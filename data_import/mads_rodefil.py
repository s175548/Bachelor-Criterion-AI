import torch
from semantic_segmentation.semi_supervised.generator import generator,generator1
import PIL,numpy as np
from PIL import Image
# model = generator(3)
# # model.load_state_dict(torch.load(r'E:\downloads_hpc_bachelor\exp_results\semi\spectral\model_g.pt'))
# model.load_state_dict(torch.load(r'E:\downloads_hpc_bachelor\DGAN_setup\Decent_results_100epochs\modelG.pt'))
# model.eval()
#
# for i in range(5):
#     noise = torch.randn(3, 100, 1, 1)
#     output = model(noise)
#     test_noise = Image.fromarray( (noise[0].reshape(10,10).numpy()*255).astype(np.uint8) )
#     Image._show(test_noise)
#     test = Image.fromarray( np.transpose((output[0].detach().numpy()*255).astype(np.uint8),(1,2,0)) )
#     Image._show(test)


model = generator1(3)
model.load_state_dict(torch.load(r'E:\downloads_hpc_bachelor\exp_results\old_experiments\semi_supervised_correct_loss\lr_exp\lr\001\model_g.pt'))
model.eval()
for i in range(4):
    noise = torch.rand([3,50*50]).uniform_()
    output = model(noise)
    test_noise = Image.fromarray( (noise[0].reshape(50,50).numpy()*255).astype(np.uint8) )
    Image._show(test_noise)
    test = Image.fromarray( np.transpose((output[0].detach().numpy()*255).astype(np.uint8),(1,2,0)) )
    Image._show(test)
    print(torch.sum(output))

### LINE 220 Training_win_semi
# loss_labeled_test1 = criterion_d_test1(pred_labeled, labels)
# loss_labeled_test2 = criterion_d_test2(pred_labeled, labels)
#
# if 2 in np.unique(labels.cpu().numpy()):
#     for i in range(4):
#         image_test = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
#         image_test = Image.fromarray((image_test / np.max(image_test) * 255).astype('uint8'))
#         Image._show(image_test)
#         mask_test = Image.fromarray((labels[0].cpu().numpy() * 125).astype('uint8'))
#         Image._show(mask_test)
# save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\tif_images'
# import numpy as np,os
# from PIL import Image
# for i in range(86):
#     test = np.zeros((2048,2048,3))
#     im_pil = Image.fromarray( (test* 255).astype(np.uint8))
#     im_pil.save(os.path.join(save_path, str(i) + "_mask.png"))

#     self.binary_class_dictionary = self.generate_binary_class_dictionary()
#
#
# def generate_binary_class_dictionary(self):
#         """     All categories found in metadata_csv are turned into dictionary, such that that can get a binary output (0: good, 1: defect) by parsing the category to the dict
#                 self.binary_class_dictionary[ self.metadata_csv[0,0] ] will return the binary value of the first datapoint.
#         """
#         binary_dict = {}
#         for ele in np.unique(self.metadata_csv[:, 0]):
#             if "good" in ele.lower():
#                 binary_dict[ele] = 0
#             else:
#                 binary_dict[ele] = 1
#         return binary_dict
#
#
#
#     def read_segmentation_file(self,filename):
#         """     Helper function, that simply opens segmentation file, draws a contour from this.
#                 Output: Segmentation retrieved from filename
#         """
#         fh = open(filename, "r")
#         try:
#             file_content = fh.read()
#             seg = json.loads(file_content)
#             segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
#             return segmentation
#         finally:
#             fh.close()
#
#             np.where(np.array([numb in dataloader.valid_annotations for numb in list(range(691))]) == False)[0] # Get index of invalid masks
#             np.where(np.array(dataloader.visibility_score) == 3)[0] #
#             np.sort(np.array(list(np.where(np.array(dataloader.visibility_score) == 3)[0]) + list(
#                 np.where(np.array(dataloader.visibility_score) == 2)[0])))



from tqdm import tqdm
import random
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
from torch.utils import data
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
import torch,cv2
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
#import os
import numpy as np
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et

from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
import PIL
import random
import pickle
import matplotlib.pyplot as plt

# model = deeplabv3_resnet101()
# model.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
# model.aux_classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
# device = torch.device('cpu')

# file = open('color_dict_multi.txt', 'w')
# file.write(str(color_dict))
# file.close()
#
# file = open('target_dict_multi.txt', 'w')
# file.write(str(target_dict))
# file.close()
#
# file = open('annotations_dict_multi.txt', 'w')
# file.write(str(annotations_dict))
# file.close()

# file_test = open('color_dict.txt', 'r')
# test = file_test.read()
# diction = ast.literal_eval(test)
PATH = r'E:\downloads_hpc_bachelor\exp_results\lr_INITIAL EXP LR V7_8_oktober\lr\0001\DeepLab_lr_exp0.0001.pt'

# Save


# Load
# model = Net()
# model.load_state_dict(torch.load(PATH))
# model.eval()
#
#
# denorm = Denormalize(mean=[0.485, 0.456, 0.406],
#                      std=[0.229, 0.224, 0.225])
# import os
# #TEST MODEL
# if __name__ == "__main__":
#     binary = False
#     batch_size = 16
#     val_batch_size = 4  # 4
#
#     path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\train'
#     path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\val'
#
#     data_loader = DataLoader()
#
#     labels = ['Piega', 'Verruca', 'Puntura insetto', 'Background']
#
#     file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] != "k"])
#     N_files = len(file_names_train)
#     shuffled_index = np.random.permutation(len(file_names_train))
#     file_names_train = file_names_train[shuffled_index]
#     file_names_train = file_names_train[file_names_train != ".DS_S"]
#
#     file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] != "k"])
#     N_files = len(file_names_val)
#     shuffled_index = np.random.permutation(len(file_names_val))
#     file_names_val = file_names_val[shuffled_index]
#     file_names_val = file_names_val[file_names_val != ".DS_S"]
#
#     transform_function = et.ExtCompose([et.ExtEnhanceContrast(), et.ExtRandomCrop((256, 256)), et.ExtToTensor(),
#                                         et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
#     if binary:
#         color_dict = data_loader.color_dict_binary
#         target_dict = data_loader.get_target_dict()
#         annotations_dict = data_loader.annotations_dict
#
#     else:
#         color_dict = data_loader.color_dict
#         target_dict = data_loader.get_target_dict(labels)
#         annotations_dict = data_loader.annotations_dict
#
#     train_dst = LeatherData(path_mask=path_train, path_img=path_train, list_of_filenames=file_names_train,
#                             transform=transform_function, color_dict=color_dict, target_dict=target_dict)
#     val_dst = LeatherData(path_mask=path_val, path_img=path_val, list_of_filenames=file_names_val,
#                           transform=transform_function, color_dict=color_dict, target_dict=target_dict)
#
#     train_loader = data.DataLoader(
#         train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = data.DataLoader(
#         val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)
#
#     train_img = []
#     for i in range(10):
#         train_img.append(train_dst.__getitem__(i))
#
#     model_path = r'E:\downloads_hpc_bachelor\exp_results\lr_INITIAL EXP LR V7_8_oktober\lr\0001\DeepLab_lr_exp0.0001.pt'
#     model_dict_parameters = {'model_pre_class': {'pretrained': True, 'num_classes': 21, 'requires_grad': False},
#                              'model_pre_full': {'pretrained': True, 'num_classes': 21, 'requires_grad': True},
#                              'model_full': {'pretrained': False, 'num_classes': 2, 'requires_grad': True}}
#     model_dict = {}
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # for model_name in ['model_pre_full']:
#     #     model = deeplabv3_resnet101(pretrained=model_dict_parameters[model_name]['pretrained'], progress=True,
#     #                                 num_classes=model_dict_parameters[model_name]['num_classes'], aux_loss=None)
#     #     for parameter in model.classifier.parameters():
#     #         parameter.requires_grad_(requires_grad=model_dict_parameters[model_name]['requires_grad'])
#     default_scope = True
#     model='DeepLab'
#     model_name = 'model_pre_full'
#     model_dict[model] = deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
#     if default_scope:
#         grad_check(model_dict[model])
#         model_dict[model_name] = model
#     lr = 0.001
#     checkpoint = torch.load(model_path.format(lr), map_location=torch.device('cpu'))
#     for model_name, model in model_dict.items():
#         model.load_state_dict(checkpoint["model_state"])
#         model = nn.DataParallel(model)
#         model.to(device)
#         del checkpoint
#         print("Model restored")
#
#         model.eval()
#         for i in range(len(train_img)):
#             image = train_img[i][0].unsqueeze(0)
#             image = image.to(device, dtype=torch.float32)
#
#             output = model(image)['out']
#             pred = output.detach().max(dim=1)[1].cpu().numpy()
#             target = train_img[i][1].cpu().numpy()
#             image = (denorm(train_img[i][0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
#             cv2.imshow('inp',np.array(np.transpose(np.squeeze(train_img[i][0]),(1,2,0))))
#             cv2.imshow('pred', np.transpose(( (~ (pred-1) * (-255) ) ).astype(np.uint8),(1,2,0)))
#             cv2.imshow('image', image)
#             cv2.imshow('target', target)
#             cv2.waitKey(0)
#             print(image.shape)
#
#TEST MODEL END

# N_epochs = 2
# train_loss_values = [0,2]
# validation_loss_values = [2,2]
#
# plt.plot(range(N_epochs), train_loss_values, '-o')
# plt.title('Train Loss')
# plt.xlabel('N_epochs')
# plt.ylabel('Loss')
# plt.savefig(os.path.join(r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\model_pre_full','train_test'), format='jpg')
# plt.close()
# plt.plot(range(N_epochs), validation_loss_values, '-o')
# plt.title('Validation Loss')
# plt.xlabel('N_epochs')
# plt.ylabel('Loss')
# plt.savefig(os.path.join(r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\model_pre_full','val_test'), format='jpg')
# plt.close()

#EXPERIMENT WITH BINARY TICK CLASS
#training(n_classes=1, model='DeepLab', load_models=False, model_path=path_model,train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst,save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict,annotations_dict=annotations_dict,exp_description='tick')
#EXPERIMENT WITH MULTI CLASS
# num_classes=2
# output_stride=16
# save_val_results=False
# total_itrs=1000#1000
# #lr=0.01,0.001,0.0001
# lr_policy='step'
# step_size=10000
# batch_size= 16 # 16
# val_batch_size= 4 #4
# loss_type="cross_entropy"
# weight_decay=1e-4
# random_seed=1
# val_interval= 55 # 55
# vis_num_samples= 2 #2
# enable_vis=True
# N_epochs= 100 # 240 #Helst mange

#training(n_classes=3, model='DeepLab', load_models=False, model_path=path_model,train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst,save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict,annotations_dict=annotations_dict,'multi')

# num_classes=2
# output_stride=16
# save_val_results=False
# total_itrs=1000#1000
# #lr=0.01,0.001,0.0001
# lr_policy='step'
# step_size=10000
# batch_size= 16 # 16
# val_batch_size= 4 #4
# loss_type="cross_entropy"
# weight_decay=1e-4
# random_seed=1
# val_interval= 55 # 55
# vis_num_samples= 2 #2
# enable_vis=True
# N_epochs= 100 # 240 #Helst mange
# import argparse,os
# save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt'
# parser = argparse.ArgumentParser(description='Take learning rate parameter')
# parser.add_argument('parameter choice', metavar='lr', type=float, nargs='+', help='a parameter for the training loop')
# parser.add_argument('folder name', metavar='folder', type=str, nargs='+', help='a save folder for the training loop')
# args = vars(parser.parse_args())
# print(save_path,'old')
# save_folder = args['folder name'][0]
# save_path = os.path.join(save_path, save_folder)
# print(save_path,'new')
# lr = args['parameter choice'][0]
# print(args['parameter choice'][0], " this is the chosen parameter")
# save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\slet'
# model_name = 'DeepLab'
# exp_description = 'Test'
# lr = 0.01
# import numpy as np,os
# import matplotlib.pyplot as plt
# plt.plot(np.linspace(1,10),np.arange(len(np.linspace(1,10))))
# plt.title('Validation Loss')
# plt.xlabel('N_epochs')
# plt.ylabel('Loss')
# plt.show()
# plt.savefig(os.path.join(save_path,exp_description+(str(lr))+'_val_loss'),format='png')
# plt.close()