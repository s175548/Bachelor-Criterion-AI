import sys,os
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')

from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
import argparse,json,ast

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

HPC =True
Villads=False
binary=False
model_name = ''
optimizer = ''
exp_descrip = ''
train_scope = ''
if __name__ == "__main__":
    if HPC:
        save_path = r'/work3/s173934/Bachelorprojekt/exp_results'
        path_model = r'/work3/s173934/Bachelorprojekt/'

        parser = argparse.ArgumentParser(description='Take parameters')
        parser.add_argument('learning rate', metavar='lr', type=float, nargs='+',help='a parameter for the training loop')
        parser.add_argument('model name', metavar='optimizer', type=str, nargs='+',help='choose either MobileNet or DeepLab')
        parser.add_argument('optimizer name', metavar='model', type=str, nargs='+',help='choose either MobileNet or DeepLab')
        parser.add_argument('train scope', metavar='scope', type=bool, nargs='+',help='train whole model or only classifier')
        parser.add_argument('experiment description', metavar='description', type=str, nargs='+',help='enter description')
        parser.add_argument('folder name', metavar='folder', type=str, nargs='+',help='a save folder for the training loop')
        parser.add_argument('binary_setup',metavar='binary setup', default=True, action='store_false', help='Bool type')

        # parser.add_argument('binary_setup',default=True, metavar='setup', type=bool,action='store_false', nargs='+', help='binary or multiclass')
        args = vars(parser.parse_args())

        lr = args['learning rate'][0]
        optimizer = args['optimizer name'][0]
        train_scope = args['train scope'][0]
        model_name = args['model name'][0]
        exp_descrip = args['experiment description'][0]
        save_folder = args['folder name'][0]
        binary = args['binary_setup'][0]
        print("train_scope: ", train_scope)
        print("save folder: ",save_folder)
        print("binary: ", binary)
        save_path = os.path.join(save_path, save_folder)

        if binary:
            print('binary training set loading')
            path_train = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_binary_vis_2_and_3/train'
            path_val = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_binary_vis_2_and_3/val'
        else:
            print('multi training set loading')
            path_train = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_vis_2_and_3/train'
            path_val = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_vis_2_and_3/val'
        path2 = r'/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/outfile.jpg'
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'



    elif Villads:
        path_img = path_mask = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data'
        path_original_data = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
        metadata_path=save_path = r'samples/model_comparison.csv'
        path_model = save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor '
        lr = 0.01
        path_meta_data = r'samples/model_comparison.csv'


    else:
        save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\slet'
        path_model = save_path
        path_original_data = r'C:\Users\Mads-\Desktop\leather_patches'
        if binary:
            path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_binary_vis_2_and_3\train'
            path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_binary_vis_2_and_3\val'
        else:
            path_train = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\train'
            path_val = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi_vis_2_and_3\val'
        path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'
        lr = 0.01
        path_meta_data = r'samples/model_comparison.csv'

    # path_img = path_mask = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data'
    # data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
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

    transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtRandomCrop((256,256)),et.ExtToTensor(),et.ExtNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    if binary:
        print('binary class dicts loading')
        color_dict = data_loader.color_dict_binary
        target_dict = data_loader.get_target_dict()
        annotations_dict = data_loader.annotations_dict

    else:
        print('multi class dicts loading')
        color_dict= data_loader.color_dict
        target_dict=data_loader.get_target_dict(labels)
        annotations_dict=data_loader.annotations_dict

    
    train_dst = LeatherData(path_mask=path_train,path_img=path_train,list_of_filenames=file_names_train,
                            transform=transform_function,color_dict=color_dict,target_dict=target_dict)
    val_dst = LeatherData(path_mask=path_val, path_img=path_val,list_of_filenames=file_names_val,
                          transform=transform_function,color_dict=color_dict,target_dict=target_dict)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)

    train_img = []
    for i in range(5):
        train_img.append(train_dst.__getitem__(i))



    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))
    if model_name == '':
        print('default',model_name)
        model_name = 'MobileNet'
        #model_name =
    if optimizer == '':
        print('default', optimizer)
        optimizer = 'SGD'
    if exp_descrip == '':
        print('default', exp_descrip)
        exp_descrip = 'no_decrip'
    if train_scope == '':
        print('default', train_scope)
        train_scope = True
    #training(n_classes=1, model="MobileNet", load_models=False, model_path=path_model,train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst,save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict,annotations_dict=annotations_dict,exp_description='tick')
    print("binary: ",binary)
    training(n_classes=3, model=model_name, load_models=False, model_path=path_model,train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst,save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict,annotations_dict=annotations_dict,exp_description = exp_descrip,optim=optimizer,default_scope = train_scope)
