import sys,os
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')

from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData

HPC = True
if __name__ == "__main__":
    if HPC:
        save_path = r'/zhome/87/9/127623/BachelorProject/'
        path_model = r'/work3/s173934/Bachelorprojekt/'
        path_mask = r'/work3/s173934/Bachelorprojekt/cropped_data/mask'
        path_img = r'/work3/s173934/Bachelorprojekt/cropped_data/img'
        path2 = r'/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/outfile.jpg'
    else:
        save_path = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt'
        path_model = os.getcwd()
        path_mask = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data\mask'
        path_img = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\cropped_data\img'
        path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg'



    file_names = np.array([image_name[:-4] for image_name in os.listdir(path_img) if image_name[:-4] !=".DS_S"])
    N_files=len(file_names)
    shuffled_index=np.random.permutation(len(file_names))
    file_names_img=file_names[shuffled_index]
    file_names=file_names[file_names != ".DS_S"]

    transform_function = et.ExtCompose([et.ExtTransformLabel(),
                    et.ExtToTensor(),
                    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),])



    train_dst = LeatherData(path_mask=path_mask,path_img=path_img,list_of_filenames=file_names[:round(N_files*0.80)], transform=transform_function)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img,list_of_filenames=file_names[round(N_files*0.80):], transform=transform_function)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=4)

    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))

    pass
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        image = images
        label = labels

    training(['model_pre_full'],path2=path2,val_loader=val_loader,train_loader=train_loader,train_dst=train_dst, val_dst=val_dst,model_path=path_model,save_path=save_path)