import sys,os
sys.path.append('/zhome/87/9/127623/BachelorProject/cropped_data/Bachelor-Criterion-AI')


from semantic_segmentation.semi_supervised.WGAN_train_semi import training, batch_size,val_batch_size
from torch.utils import data
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from semantic_segmentation.semi_supervised.helpful_functions import get_data_loaders_unlabelled
from data_import.data_loader import DataLoader
import argparse,json,ast,numpy as np
from PIL import Image
from torchvision import transforms
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
    HPC = True
    SIZE =256
    semi_supervised = True
    binary = True
    lr   = 0.00005  #
    lr_g = 0.00005
    exp_descrip = optimizer = model_name = ''
    train_scope = ''
    if HPC:
        save_path = r'/work3/s173934/Bachelorprojekt/exp_results'
        path_model = r'/work3/s173934/Bachelorprojekt/'

        parser = argparse.ArgumentParser(description='Take parameters')
        parser.add_argument('learning rate d', metavar='lr', type=float, nargs='+',help='a parameter for the training loop')
        parser.add_argument('learning rate g', metavar='lr', type=float, nargs='+',help='a parameter for the training loop')
        parser.add_argument('experiment description', metavar='description', type=str, nargs='+',help='enter description')
        parser.add_argument('folder name', metavar='folder', type=str, nargs='+',help='a save folder for the training loop')

        args = vars(parser.parse_args())

        lr = args['learning rate d'][0]
        lr_g = args['learning rate g'][0]
        exp_descrip = args['experiment description'][0]
        save_folder = args['folder name'][0]
        optimizer = 'SGD'
        train_scope = False
        model_name = 'DeepLab'
        binary = True
        semi_supervised = True
        save_path = os.path.join(save_path, save_folder)

        if binary:
            path_train = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/train'
            path_val = r'/work3/s173934/Bachelorprojekt/data_binary_all_classes/data_binary_all_classes/val'
            dataset_path_ul = r'/work3/s173934/Bachelorprojekt/all'

        path2 = r'/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/outfile.jpg'
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'

    else:
        save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\slet'
        path_model = save_path
        path_original_data = r'C:\Users\Mads-\Desktop\leather_patches'
        if binary:
            path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_vis_2_and_3_recreate\train'
            path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\data_binary_vis_2_and_3_recreate\val'
            dataset_path_ul = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\trained_models'
        path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'
        path_meta_data = r'samples/model_comparison.csv'

    data_loader = DataLoader(data_path=path_original_data,metadata_path=path_meta_data)
    labels=['Piega', 'Verruca', 'Puntura insetto','Background']

    file_names_train = np.array([image_name[:-4] for image_name in os.listdir(path_train) if image_name[-5] !="k"])
    N_files=len(file_names_train)
    shuffled_index=np.random.permutation(len(file_names_train))
    file_names_train=file_names_train[shuffled_index]
    file_names_train=file_names_train[file_names_train != ".DS_S"]

    file_names_val = np.array([image_name[:-4] for image_name in os.listdir(path_val) if image_name[-5] !="k"])
    N_files=len(file_names_val)
    shuffled_index=np.random.permutation(len(file_names_val))
    file_names_val=file_names_val[shuffled_index]
    file_names_val=file_names_val[file_names_val != ".DS_S"]

    # #FOR EXTENDED DATASET EXPERIMENT
    transform_function = et.ExtCompose([et.ExtRandomHorizontalFlip(p=0.5),et.ExtRandomCrop(size=SIZE),et.ExtEnhanceContrast(),et.ExtRandomVerticalFlip(p=0.5),et.ExtToTensor(),
                                        et.ExtNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    if binary:
        color_dict = data_loader.color_dict_binary
        target_dict = data_loader.get_target_dict()
        annotations_dict = data_loader.annotations_dict

    train_dst = LeatherData(path_mask=path_train,path_img=path_train,list_of_filenames=file_names_train,
                            transform=transform_function,color_dict=color_dict,target_dict=target_dict)
    val_dst = LeatherData(path_mask=path_val, path_img=path_val,list_of_filenames=file_names_val,
                          transform=transform_function,color_dict=color_dict,target_dict=target_dict)

    train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=0)
    # Load dataloader for unlabelled data:
    trainloader_nl, _ = get_data_loaders_unlabelled(binary, path_original_data, path_meta_data, dataset_path_ul,batch_size,size = SIZE)


    train_img = []
    for i in range(2):
        train_img.append(train_dst.__getitem__(i))


    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))
    if model_name == '':
        model_name = 'DeepLab'
        #model_name =
    if optimizer == '':
        optimizer = 'Adam'
    if exp_descrip == '':
        exp_descrip = 'no_decrip'
    if train_scope == '':
        train_scope = True

    #training(n_classes=1, model="MobileNet", load_models=False, model_path=path_model,train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst,save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict,annotations_dict=annotations_dict,exp_description='tick')
    training(n_classes=1, model=model_name, load_models=False, model_path=path_model, train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst, save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict, annotations_dict=annotations_dict, exp_description = exp_descrip, optim=optimizer, default_scope = train_scope, semi_supervised=semi_supervised,
                 trainloader_nl=trainloader_nl,lr_g = lr_g)