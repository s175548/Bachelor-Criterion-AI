from torch.utils import data
import os, pickle
import numpy as np
from PIL import Image
import torch
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from object_detect.generate_bbox import LeatherData_BB
import object_detect.helper.utils as utils

transform_function = et.ExtCompose([et.ExtEnhanceContrast(),et.ExtScale(512),et.ExtToTensor()])
if __name__ == '__main__':
    device = torch.device('cpu')

    path_mask = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\mask'
    path_img = r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\img'

    batch_size = 1
    val_batch_size = 4

    visibility_scores = [3]

    if type(visibility_scores) == list:
        with open(
                r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg',
                'rb') as fp:
            itemlist = np.array(pickle.load(fp))

    file_names = np.array([img[:-4] for img in os.listdir(path_img)])
    itemlist = itemlist[file_names.astype(np.uint8)]
    file_names = np.sort(file_names)[itemlist == 3]
    N_files = len(file_names)
    shuffled_index = np.random.permutation(len(file_names))
    file_names_img = file_names[shuffled_index]
    file_names = file_names[file_names != ".DS_S"]

    scale = 512
    # Define dataloaders
    train_dst = LeatherData_BB(path_mask=path_mask, path_img=path_img,
                               list_of_filenames=file_names[:50], scale=scale, transform=transform_function)
    val_dst = LeatherData_BB(path_mask=path_mask, path_img=path_img,
                             list_of_filenames=file_names[130:140], scale=scale, transform=transform_function)

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))

    for (mask, bmask, bbox) in train_loader:
        Image._show(Image.fromarray(mask[0]))
        Image._show(Image.fromarray(bmask[0]))
        jo = 0
        joo = 0