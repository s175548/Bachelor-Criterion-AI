import sys
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')

from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
import argparse,os

HPC = True
if __name__ == "__main__":
    if HPC:
        save_path = r'/zhome/87/9/127623/BachelorProject/'
        path_model = r'/work3/s173934/Bachelorprojekt/'
        path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
        path_img = r'/work3/s173934/Bachelorprojekt/cropped_data_tickbite_vis_2_and_3'
        path2 = r'/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/outfile.jpg'
        path_original_data = r'/work3/s173934/Bachelorprojekt/leather_patches'

        parser = argparse.ArgumentParser(description='Take learning rate parameter')
        parser.add_argument('learning rate', metavar='lr', type=float, nargs='+',help='a learning rate for the training loop')
        args = vars(parser.parse_args())
        lr = args['learning rate'][0]
        print(args['learning rate'][0], " this is the learning_rate")

    else:
        save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt'
        path_model = save_path
        path_original_data = r'C:\Users\Mads-\Desktop\leather_patches'
        path_img = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_tickbite_vis_2_and_3'
        path_mask = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data_tickbite_vis_2_and_3'
        path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'
        lr = 0.01

    # path_img = path_mask = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data'
    # data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
    data_loader = DataLoader(data_path=path_original_data)

    #path_img = path_mask = '/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/cropped_data'
    #data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')

    labels=['Piega', 'Verruca', 'Puntura insetto','Background']


#    data_loader = DataLoader()

    file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[-5] !="k"])
    N_files=len(file_names)
    shuffled_index=np.random.permutation(len(file_names))
    file_names_img=file_names[shuffled_index]
    file_names=file_names[file_names != ".DS_S"]

    transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtRandomCrop((256,256)),
                    et.ExtToTensor(),
                    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),])


    color_dict=data_loader.color_dict

    target_dict=data_loader.get_target_dict()

    train_dst = LeatherData(path_mask=path_mask,path_img=path_img,list_of_filenames=file_names[:round(N_files*0.80)],
                            transform=transform_function,color_dict=color_dict,target_dict=target_dict)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img,list_of_filenames=file_names[round(N_files*0.80):],
                          transform=transform_function,color_dict=color_dict,target_dict=target_dict)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)

    train_img = []
    for i in range(2):
        train_img.append(train_dst.__getitem__(i))



    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

  #  training(['model_pre_full'],path2=path2,val_loader=val_loader,train_loader=train_loader,train_dst=train_dst, val_dst=val_dst,model_path=path_model,save_path=save_path,lr=lr,train_images=train_img)
