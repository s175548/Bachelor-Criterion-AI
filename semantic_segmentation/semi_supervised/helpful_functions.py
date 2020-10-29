import sys, os

sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')

from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
import argparse, json, ast
from PIL import Image
from torchvision import transforms
from semantic_segmentation.DeepLabV3.network.utils import randomCrop, pad
from semantic_segmentation.DeepLabV3.Training_windows import my_def_collate


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

model_name = ''
optimizer = ''
exp_descrip = ''
train_scope = ''
SIZE = 512
def get_paths(binary=True,HPC=True,Villads=False,Johannes=False):
    if HPC:
        save_path = r'/work3/s173934/Bachelorprojekt/exp_results'
        path_model = r'/work3/s173934/Bachelorprojekt/'

        parser = argparse.ArgumentParser(description='Take parameters')
        parser.add_argument('learning rate', metavar='lr',default=0.1, type=float, nargs='+',help='a parameter for the training loop')
        parser.add_argument('model name', metavar='optimizer', type=str,default="MobileNet", nargs='+',help='choose either MobileNet or DeepLab')
        parser.add_argument('optimizer name', metavar='model', type=str,default = "SGD", nargs='+',help='choose either MobileNet or DeepLab')
        parser.add_argument('train scope', default=True, type=boolean_string, nargs='+',help='train whole model or only classifier')
        parser.add_argument('experiment description', metavar='description',default='semi_supervised', type=str, nargs='+',help='enter description')
        parser.add_argument('folder name', metavar='folder', type=str,default = 'semi_supervised', nargs='+',help='a save folder for the training loop')
        parser.add_argument('binary_setup', default=True, type=boolean_string, nargs='+', help='binary or multiclass')
        args = vars(parser.parse_args())

        lr = args['learning rate'][0]
        optimizer = args['optimizer name'][0]
        train_scope = args['train scope'][0]
        model_name = args['model name'][0]
        exp_descrip = args['experiment description'][0]
        save_folder = args['folder name'][0]
        binary = args['binary_setup'][0]
        print("train_scope: ", train_scope)
        print("save folder: ", save_folder)
        print("binary: ", binary)
        save_path = os.path.join(save_path, save_folder)

        if binary:
            path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train'
            path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'
            dataset_path_ul = r'/work3/s173934/Bachelorprojekt/all'

        else:
            path_train = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_vis_2_and_3/train'
            path_val = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_vis_2_and_3/val'
        path2 = r'/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/outfile.jpg'
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'



    elif Villads:
        path_img = path_mask = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data'
        path_original_data = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
        metadata_path = save_path = r'samples/model_comparison.csv'
        path_model = save_path = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor '
        lr = 0.01
        path_meta_data = r'samples/model_comparison.csv'


    else:
        save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\slet'
        path_model = save_path
        path_original_data = r'C:\Users\Mads-\Desktop\leather_patches'
        #dataset_path_ul = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\tif_images'
        dataset_path_ul = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\trained_models'
        if binary:
            path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_all_classes\train'
            path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_all_classes\val'
        else:
            path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\train'
            path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\val'
        path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'
        lr = 0.01
        path_meta_data = r'samples/model_comparison.csv'
    return path_original_data, path_meta_data, save_path,path_model,path_train,path_val,dataset_path_ul


def get_data_loaders(binary,path_original_data,path_meta_data,dataset_path_train,dataset_path_val,batch_size=16,val_batch_size=4):
    data_loader = DataLoader(data_path=path_original_data, metadata_path=path_meta_data)
    labels = ['Piega', 'Verruca', 'Puntura insetto', 'Background']

    file_names_train = np.array(
        [image_name[:-4] for image_name in os.listdir(dataset_path_train) if image_name[-5] != "k"])
    N_files = len(file_names_train)
    shuffled_index = np.random.permutation(len(file_names_train))
    file_names_train = file_names_train[shuffled_index]
    file_names_train = file_names_train[file_names_train != ".DS_S"]

    file_names_val = np.array([image_name[:-4] for image_name in os.listdir(dataset_path_val) if image_name[-5] != "k"])
    N_files = len(file_names_val)
    shuffled_index = np.random.permutation(len(file_names_val))
    file_names_val = file_names_val[shuffled_index]
    file_names_val = file_names_val[file_names_val != ".DS_S"]

    transform_function = et.ExtCompose(
        [et.ExtRandomCrop(scale=0.7, size=None),et.ExtRandomCrop(size=SIZE,pad_if_needed=True),et.ExtRandomRotation(random.randint(0,359)),et.ExtRandomHorizontalFlip(p=0.5), et.ExtRandomVerticalFlip(p=0.5),
         et.ExtEnhanceContrast(), et.ExtToTensor(),et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_function_val = et.ExtCompose(
        [et.ExtRandomCrop(scale=0.7, size=None),et.ExtRandomCrop(size=SIZE,pad_if_needed=True),et.ExtEnhanceContrast(), et.ExtToTensor(),
         et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if binary:
        color_dict = data_loader.color_dict_binary
        target_dict = data_loader.get_target_dict()
        annotations_dict = data_loader.annotations_dict

    else:
        color_dict = data_loader.color_dict
        target_dict = data_loader.get_target_dict(labels)
        annotations_dict = data_loader.annotations_dict

    train_dst = LeatherData(path_mask=dataset_path_train, path_img=dataset_path_train,
                            list_of_filenames=file_names_train,
                            transform=transform_function, color_dict=color_dict, target_dict=target_dict)
    val_dst = LeatherData(path_mask=dataset_path_val, path_img=dataset_path_val, list_of_filenames=file_names_val,
                          transform=transform_function_val, color_dict=color_dict, target_dict=target_dict)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_dst, val_dst, color_dict, target_dict, annotations_dict

def get_data_loaders_unlabelled(binary,path_original_data,path_meta_data,dataset_path_unlabelled,batch_size):
    data_loader = DataLoader(data_path=path_original_data, metadata_path=path_meta_data)
    labels = ['Piega', 'Verruca', 'Puntura insetto', 'Background']

    file_names_train = np.array([image_name[:-4] for image_name in os.listdir(dataset_path_unlabelled) if image_name[-5] != "k"])
    N_files = len(file_names_train)
    shuffled_index = np.random.permutation(len(file_names_train))
    file_names_train = file_names_train[shuffled_index]
    file_names_train = file_names_train[file_names_train != ".DS_S"]

    transform_function = et.ExtCompose(
        [et.ExtRandomCrop(scale=0.7, size=None), et.ExtRandomCrop(size=SIZE, pad_if_needed=True),
         et.ExtRandomRotation(random.randint(0, 359)), et.ExtRandomHorizontalFlip(p=0.5),
         et.ExtRandomVerticalFlip(p=0.5),
         et.ExtEnhanceContrast(), et.ExtToTensor(),
         et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # et.ExtRandomCrop(scale=0.7)
    if binary:
        color_dict = data_loader.color_dict_binary
        target_dict = data_loader.get_target_dict()
        annotations_dict = data_loader.annotations_dict

    else:
        color_dict = data_loader.color_dict
        target_dict = data_loader.get_target_dict(labels)
        annotations_dict = data_loader.annotations_dict

    trainloader_nl_dst = LeatherData(path_mask=dataset_path_unlabelled, path_img=dataset_path_unlabelled,list_of_filenames=file_names_train,transform=transform_function, color_dict=color_dict, target_dict=target_dict)
    trainloader_nl = data.DataLoader(trainloader_nl_dst, batch_size=batch_size, shuffle=True, num_workers=4)
    return trainloader_nl, trainloader_nl_dst