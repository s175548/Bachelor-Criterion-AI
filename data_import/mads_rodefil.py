#
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



# from tqdm import tqdm
# import random
# import numpy as np
#from semantic_segmentation.DeepLabV3.dataset_class import LeatherData

from torch.utils import data
#from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics
#from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
# from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
import torch,cv2
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import os
import numpy as np
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et

# from semantic_segmentation.DeepLabV3.Training_windows import *
# from semantic_segmentation.DeepLabV3.Training_windows import *
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


#TEST MODEL
# if __name__ == "__main__":
#     batch_size = 16
#     val_batch_size = 4  # 4
#
#     save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt'
#     path_model = os.getcwd()
#     path_mask = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_28_09\mask'
#     path_img = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_28_09\img'
#     path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'
#
#     file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[:-4] != ".DS_S"])
#     N_files = len(file_names)
#     shuffled_index = np.random.permutation(len(file_names))
#     file_names_img = file_names[shuffled_index]
#     file_names = file_names[file_names != ".DS_S"]
#
#     denorm = Denormalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
#     transform_function = et.ExtCompose([et.ExtTransformLabel(),
#                                         et.ExtToTensor(),
#                                         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                                         std=[0.229, 0.224, 0.225]), ])
#
#     train_dst = LeatherData(path_mask=path_mask, path_img=path_img,
#                             list_of_filenames=file_names[:round(N_files * 0.80)],
#                             transform=transform_function)
#     val_dst = LeatherData(path_mask=path_mask, path_img=path_img, list_of_filenames=file_names[round(N_files * 0.80):],
#                           transform=transform_function)
#     train_loader = data.DataLoader(
#         train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = data.DataLoader(
#         val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)
#
#     train_img = []
#     for i in range(20):
#         train_img.append(train_dst.__getitem__(random.randint(0,500)))
#
#     model_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Downloads_hpc\Bachelor-Criterion-AImodel_pre_full{}.pt'
#     model_dict_parameters = {'model_pre_class': {'pretrained': True, 'num_classes': 21, 'requires_grad': False},
#                              'model_pre_full': {'pretrained': True, 'num_classes': 21, 'requires_grad': True},
#                              'model_full': {'pretrained': False, 'num_classes': 2, 'requires_grad': True}}
#     model_dict = {}
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     for lr in ['0.01','0.001','0.001']:
#         for model_name in ['model_pre_full']:
#             model = deeplabv3_resnet101(pretrained=model_dict_parameters[model_name]['pretrained'], progress=True,
#                                         num_classes=model_dict_parameters[model_name]['num_classes'], aux_loss=None)
#             for parameter in model.classifier.parameters():
#                 parameter.requires_grad_(requires_grad=model_dict_parameters[model_name]['requires_grad'])
#
#             if model_dict_parameters[model_name]['num_classes'] == 21:
#                 model.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
#                 model.aux_classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
#             model_dict[model_name] = model
#
#         checkpoint = torch.load(model_path.format(lr), map_location=torch.device('cpu'))
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