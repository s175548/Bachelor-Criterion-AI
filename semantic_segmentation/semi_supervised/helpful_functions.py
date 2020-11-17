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
        parser.add_argument('semi_supervised', default=True, type=boolean_string, nargs='+', help='semi_supervised')

        args = vars(parser.parse_args())

        lr = args['learning rate'][0]
        optimizer = args['optimizer name'][0]
        train_scope = args['train scope'][0]
        model_name = args['model name'][0]
        exp_descrip = args['experiment description'][0]
        save_folder = args['folder name'][0]
        semi_supervised = args['semi_supervised'][0]
        binary = args['binary_setup'][0]
        print("train_scope: ", train_scope)
        print("save folder: ", save_folder)
        print("binary: ", binary)
        save_path = os.path.join(save_path, save_folder)

        if binary:
            path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_vis_2_and_3/train'
            path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_vis_2_and_3/val'
            dataset_path_ul = r'/work3/s173934/Bachelorprojekt/all'

        else:
            path_train = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_vis_2_and_3/train'
            path_val = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_vis_2_and_3/val'
        path2 = r'/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/outfile.jpg'
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'

        return path_original_data, path_meta_data, save_path, path_model, path_train, path_val, dataset_path_ul, model_name,exp_descrip,semi_supervised,lr


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
            #path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_all_classes\train'
            #path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_all_classes\val'
            path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_vis_2_and_3\train'
            path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_vis_2_and_3\val'
        else:
            path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\train'
            path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\val'
        path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'
        lr = 0.01
        path_meta_data = r'samples/model_comparison.csv'
    return path_original_data, path_meta_data, save_path,path_model,path_train,path_val,dataset_path_ul


def get_data_loaders(binary,path_original_data,path_meta_data,dataset_path_train,dataset_path_val,batch_size=16,val_batch_size=4,size=512):
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

    # transform_function = et.ExtCompose([et.ExtCenterCrop(size=SIZE),et.ExtRandomHorizontalFlip(p=0.5), et.ExtRandomVerticalFlip(p=0.5),
    #      et.ExtEnhanceContrast(), et.ExtToTensor(),et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_function = et.ExtCompose([et.ExtRandomCrop(scale=0.7, size=None),et.ExtRandomCrop(size=size,pad_if_needed=True),et.ExtRandomHorizontalFlip(p=0.5), et.ExtRandomVerticalFlip(p=0.5),
         et.ExtEnhanceContrast(), et.ExtToTensor(),et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_function_val = et.ExtCompose([
        et.ExtRandomCrop(scale=0.7, size=None),et.ExtRandomCrop(size=size,pad_if_needed=True),
        et.ExtEnhanceContrast(), et.ExtToTensor(),et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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

def get_data_loaders_unlabelled(binary,path_original_data,path_meta_data,dataset_path_unlabelled,batch_size,size=512):
    data_loader = DataLoader(data_path=path_original_data, metadata_path=path_meta_data)
    labels = ['Piega', 'Verruca', 'Puntura insetto', 'Background']

    file_names_train = np.array([image_name[:-4] for image_name in os.listdir(dataset_path_unlabelled) if image_name[-5] != "k"])
    N_files = len(file_names_train)
    shuffled_index = np.random.permutation(len(file_names_train))
    file_names_train = file_names_train[shuffled_index]
    file_names_train = file_names_train[file_names_train != ".DS_S"]

    transform_function = et.ExtCompose(
        [et.ExtRandomCrop(size=size, pad_if_needed=True),
         et.ExtRandomHorizontalFlip(p=0.5),
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

    trainloader_nl_dst = LeatherData(path_mask=dataset_path_unlabelled, path_img=dataset_path_unlabelled,list_of_filenames=file_names_train,transform=transform_function, color_dict=color_dict, target_dict=target_dict,unlabelled=True)
    trainloader_nl = data.DataLoader(trainloader_nl_dst, batch_size=batch_size, shuffle=True, num_workers=4)
    return trainloader_nl, trainloader_nl_dst

def add_spectral(model):
    count = 0
    for name_layer, layer in enumerate(model.backbone):
        if (isinstance(model.backbone[layer], nn.Conv2d)):
            cur_lay = model.backbone[layer]
            kernel_size, stride, padding ,weight, bias = extract_hyperparams(cur_lay)
            model.backbone[layer] = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=cur_lay.weight.shape[1], out_channels=cur_lay.weight.shape[0],kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            count += 1
        elif (isinstance(model.backbone[layer],nn.Sequential)):
            for name_seq, layer_seq in enumerate(model.backbone[layer]):
                for key, value in model.backbone[layer][name_seq]._modules.items():
                    if (isinstance(model.backbone[layer][name_seq]._modules[key], nn.Conv2d)):
                        cur_lay = model.backbone[layer][name_seq]._modules[key]
                        kernel_size, stride, padding, weight, bias = extract_hyperparams(cur_lay)
                        model.backbone[layer][name_seq]._modules[key] = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=cur_lay.weight.shape[1],out_channels=cur_lay.weight.shape[0], kernel_size=kernel_size,padding=padding, stride=stride, bias=False))
                        count += 1
                    elif (isinstance(model.backbone[layer][name_seq]._modules[key], nn.Sequential)):
                        for name_bot_seq,layer_bot_seq in enumerate(model.backbone[layer][name_seq]._modules[key]):
                            if (isinstance(model.backbone[layer][name_seq]._modules[key][name_bot_seq], nn.Conv2d)):
                                cur_lay = model.backbone[layer][name_seq]._modules[key][name_bot_seq]
                                kernel_size, stride, padding, weight, bias = extract_hyperparams(cur_lay)
                                model.backbone[layer][name_seq]._modules[key][name_bot_seq] = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=cur_lay.weight.shape[1], out_channels=cur_lay.weight.shape[0],kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
                                count += 1
    count_clas = 0
    for name_layer, layer in enumerate(model.classifier):
        if (isinstance(model.classifier[name_layer], nn.Conv2d)):
            cur_lay = model.classifier[name_layer]
            kernel_size, stride, padding ,weight, bias = extract_hyperparams(cur_lay)
            model.classifier[name_layer] = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=cur_lay.weight.shape[1], out_channels=cur_lay.weight.shape[0],kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            count_clas += 1
        elif (isinstance(model.classifier[name_layer], nn.BatchNorm2d)):
            continue
        elif (isinstance(model.classifier[name_layer], nn.ReLU)):
            continue
        else:
            for key, value in model.classifier[name_layer]._modules.items():
                for idx,item in enumerate(model.classifier[name_layer]._modules[key]):
                    if (isinstance(model.classifier[name_layer]._modules[key][idx], nn.Sequential)):
                        for name_seq,layer_seq in enumerate(model.classifier[name_layer]._modules[key][idx]._modules):
                            if (isinstance(model.classifier[name_layer]._modules[key][idx]._modules[layer_seq],nn.Conv2d)):
                                cur_lay = model.classifier[name_layer]._modules[key][idx]._modules[layer_seq]
                                kernel_size, stride, padding, weight, bias = extract_hyperparams(cur_lay)
                                model.classifier[name_layer]._modules[key][idx]._modules[layer_seq] = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=cur_lay.weight.shape[1], out_channels=cur_lay.weight.shape[0],kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
                                count_clas += 1
                            else:
                                continue
                    elif (isinstance(model.classifier[name_layer]._modules[key][idx], nn.Conv2d)):
                        cur_lay = model.classifier[name_layer]._modules[key][idx]
                        kernel_size, stride, padding, weight, bias = extract_hyperparams(cur_lay)
                        model.classifier[name_layer]._modules[key][idx] = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=cur_lay.weight.shape[1], out_channels=cur_lay.weight.shape[0],kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
                        count_clas += 1
    return model


def extract_hyperparams(layer):
    kernel_size = layer.kernel_size[0]
    stride = layer.stride[0]
    padding = layer.padding[0]
    weight = layer.weight.unsqueeze(2) / kernel_size
    weight = torch.cat([weight for _ in range(0, kernel_size)], dim=2)
    bias = layer.bias
    return kernel_size, stride, padding ,weight, bias