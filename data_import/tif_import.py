from PIL import Image
import PIL.Image,numpy as np,os
from data_import.data_loader import DataLoader

PIL.Image.MAX_IMAGE_PIXELS = 433051264

def load_tif_as_numpy_array(path):
    im = Image.open(path)
    pixels = np.array(im,dtype=np.float32)
    return pixels

def convert_numpy_array_to_pil(array):
    return Image.fromarray(array)

def save_patches(images,path):
    for idx,img in enumerate(images):
        im_pil = Image.fromarray(img)
        im_pil.save( os.path.join(path,str(idx)+"_"+str(idx) + ".png") )

if __name__ == '__main__':
    tif_path = r'C:\Users\Mads-_uop20qq\Downloads\WALKNAPPA_VDA_04_grain_03_v.tif'
    save_path = r'E:\BachelorProj\Bachelorprojekt\cropped_tif\VDA4'
    data_loader = DataLoader()

    array = load_tif_as_numpy_array(tif_path)
    split_imgs, split_x_y, pad_split_imgs = data_loader.generate_tif_patches(array,patch_size=512*4,padding=False) # Set padding to make better image predictions

    for idx,img in enumerate(split_imgs):
        im_pil = Image.fromarray(img.astype(np.uint8))
        im_pil.save(os.path.join(save_path, str(idx)+'_grain_03' + ".png"))
    with open("{}/{}_.txt".format(save_path, 'split_x_y'), "w") as text_file:
        text_file.write(str(split_x_y))

