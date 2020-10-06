import sys,os
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')

from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
import argparse,json,ast



HPC =True
Villads=False
binary=False

if __name__ == "__main__":
    if HPC:
        save_path = r'/work3/s173934/Bachelorprojekt/exp_results'
        path_model = r'/work3/s173934/Bachelorprojekt/'
        if binary:
            # path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
            # path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
            path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_multi_vis_2_and_3/mask'
            path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3/val'
        else:
            path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_multi'
            path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_multi'
        path2 = r'/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/outfile.jpg'
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'
        path_meta_data = r'samples/model_comparison.csv'

        parser = argparse.ArgumentParser(description='Take learning rate parameter')
        parser.add_argument('parameter choice', metavar='lr', type=float, nargs='+',help='a parameter for the training loop')
        parser.add_argument('folder name', metavar='folder', type=str, nargs='+',help='a save folder for the training loop')
        args = vars(parser.parse_args())

        save_folder = args['folder name'][0]
        save_path = os.path.join(save_path,save_folder)
        lr = args['parameter choice'][0]
        print(args['parameter choice'][0], " this is the chosen parameter")

    elif Villads:
        path_img = path_mask = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data'
        path_original_data = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches'
        metadata_path=save_path = r'samples/model_comparison.csv'
        path_model = save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor '
        lr = 0.01
        path_meta_data = r'samples/model_comparison.csv'


    else:
        save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt'
        path_model = save_path
        path_original_data = r'C:\Users\Mads-\Desktop\leather_patches'
        if binary:
            path_img = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_tickbite_vis_2_and_3'
            path_mask = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_tickbite_vis_2_and_3'
        else:
            path_img = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi'
            path_mask = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_multi'
        path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'
        lr = 0.01
        path_meta_data = r'samples/model_comparison.csv'

    # path_img = path_mask = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data'
    # data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
    data_loader = DataLoader(data_path=path_original_data,metadata_path=path_meta_data)



    labels=['Piega', 'Verruca', 'Puntura insetto','Background']



    file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[-5] !="k"])
    N_files=len(file_names)
    shuffled_index=np.random.permutation(len(file_names))
    file_names_img=file_names[shuffled_index]
    file_names=file_names[file_names != ".DS_S"]

    transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtRandomCrop((256,256)),
                   et.ExtToTensor(),
                   et.ExtNormalize(mean=[0.485, 0.456, 0.406],


                                   std=[0.229, 0.224, 0.225])])

    if binary:
        color_dict = data_loader.color_dict_binary
        target_dict = data_loader.get_target_dict()
        annotations_dict = data_loader.annotations_dict

    else:
        color_dict= data_loader.color_dict
        target_dict=data_loader.get_target_dict(labels)
        annotations_dict=data_loader.annotations_dict

    
    train_dst = LeatherData(path_mask=path_mask,path_img=path_img,list_of_filenames=file_names[:round(N_files*0.80)],
                            transform=transform_function,color_dict=color_dict,target_dict=target_dict)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img,list_of_filenames=file_names[round(N_files*0.80):],
                          transform=transform_function,color_dict=color_dict,target_dict=target_dict)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)

    train_img = []
    for i in range(5):
        train_img.append(train_dst.__getitem__(i))



    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

    training(n_classes=1, model="MobileNet", load_models=False, model_path=path_model,train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst,save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict,annotations_dict=annotations_dict,exp_description='tick')

    training(n_classes=3, model='DeepLab', load_models=False, model_path=path_model,train_loader=train_loader, val_loader=val_loader, train_dst=train_dst, val_dst=val_dst,save_path=save_path, lr=lr, train_images=train_img, color_dict=color_dict, target_dict=target_dict,annotations_dict=annotations_dict,exp_description = 'multi_class')