import json, cv2, torch, os,pandas as pd,numpy as np
from PIL import Image
from torchvision import datasets, transforms
from data_import.draw_contours import draw_contours2,make_contour
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self,data_path=r'C:\Users\Mads-\Desktop\leather_patches',metadata_path = r'.samples\model_comparison.csv'):
        self.data_path = data_path
        self.metadata_path = os.path.join(data_path,metadata_path)
        self.metadata_csv = self.get_metadata(self.metadata_path)
        self.valid_annotations = self.get_empty_segmentations()
        self.visibility_score = [self.get_visibility_score( os.path.join(self.data_path,self.metadata_csv[img_idx,3] ) ) for img_idx in range(len(self.metadata_csv))]

    def get_metadata(self,metadata_path):
        """     Collect the metadata_csv file containing 7 datapoints:
                0: category; 1: path; 2: etag; 3: segmentation_path; 4: segmentation_etag; 5: model_segmentation_path; 6: model_segmentation_etag
                (All categories can for example be retrieved by self.metadata_csv[:,0])
        """
        metadata_csv = pd.read_csv(metadata_path,sep=";")
        metadata_csv.to_numpy()
        return metadata_csv.to_numpy()

    def get_visibility_score(self,filepath):
        visibility = -1
        ann = self.get_json_file_content(filepath)
        for a in ann["annotations"]:
            try:
                visibility = a["visibility"]
                break
            except KeyError:
                pass
            if a["label"].startswith("visibility_"):
                visibility = a["label"].split("_")[-1]
                break
        return int(visibility)

    def get_image_and_labels(self,images_idx):
        """     input: give index/indices of the wanted images in the dataset
                output: image(s) and mask(s) of the given index/indices
        """
        images = []
        segmentation_masks = []
        if type(images_idx) == list:
            for image_idx in images_idx:
                images.append( cv2.imread( os.path.join(self.data_path, self.metadata_csv[image_idx,1]) ))
                segmentation_masks.append( self.read_segmentation_file( os.path.join(self.data_path,self.metadata_csv[image_idx,3]) ) )
            return (images,segmentation_masks)
        else:
            return cv2.imread( os.path.join(self.data_path, self.metadata_csv[images_idx,1]) ),self.read_segmentation_file( os.path.join(self.data_path,self.metadata_csv[images_idx,3]))


    def read_segmentation_file(self,filepath):
        """     Helper function, that simply opens segmentation file, draws a contour from this.
                Output: Segmentation retrieved from filename
        """
        seg = self.get_json_file_content(filepath)
        segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
        return segmentation

    def get_json_file_content(self,filename):
        with open(filename,'r') as fh:
            file_content = fh.read()
            seg = json.loads(file_content)
        return seg

    def get_empty_segmentations(self):
        """     Some pictures in the dataset does not have proper segmentations.
                A list of all the indices of the images with correct segmentations are extracted and retunrned here.
        """
        empty = []
        for i in range(len(self.metadata_csv)):
            file_path = os.path.join(self.data_path, self.metadata_csv[i, 3])
            with open(file_path) as file:
                content = file.read()
                seg = json.loads(content)
                if seg['annotations'] == list():
                    empty.append((i,self.metadata_csv[i, 3]))
        return [i for i in range(len(self.metadata_csv)) if i not in [anno[0] for anno in empty]]

    def simple_plot_cv2(self,object,title=""):
        cv2.imshow(title,cv2.resize(object,(512,512)))
        cv2.waitKey(0)

    def plot_function(self,images, masks = np.array([])):
        """input: image(s) and mask(s)
            The function will plot image(s), and mask(s) (if given)
        """
        if len(images.shape)<4:
            self.simple_plot_cv2(images)
            if len(masks)>0:
                self.simple_plot_cv2(masks)
        else:
            for idx,image in enumerate(images):
                self.simple_plot_cv2(image)
                if len(masks)>0:
                    self.simple_plot_cv2(masks[idx])
    def generate_patches(self,img, msk,patch_size=360,print_=False):
        images = []
        masks = []
        crop_count_height = int(np.floor(img.shape[0] / patch_size))
        crop_count_width =  int(np.floor(img.shape[1] / patch_size))

        if print_:
            cv2.imshow('',img)
            cv2.waitKey(0)
        if (img.shape[0] > patch_size and img.shape[1] > patch_size):
            for i in range(crop_count_height):
                for j in range(crop_count_width):
                    image = img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                    mask = msk[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]

                    if print_:
                        cv2.imshow('image', image)
                        cv2.imshow('mask',mask)
                        cv2.waitKey(0)

                    images.append(image)
                    masks.append(mask)
        return images,masks

def to_tensor_and_normalize(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 has BGR channels, and Pillow has RGB channels, so they are transformed here
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tranformation = transforms.Compose([transforms.ToTensor(),normalize])
    a = tranformation(img2)
    #b = transforms.ToPILImage()(a)
    #b.show()
    return a
    
def test_transforms(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 has BGR channels, and Pillow has RGB channels, so they are transformed here
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.Resize(150),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.ToTensor(),normalize])
    #transforms.RandomCrop(224)
    im_pil = Image.fromarray(img2)
    transformed_sample = img_transform(im_pil)

    im_pil.show()
    transforms.ToPILImage()(transformed_sample).show()
def test_transforms_mask(mask):
    _, binary = cv2.threshold(mask*255, 225, 255, cv2.THRESH_BINARY_INV)
    im_pil = Image.fromarray(binary)
    img_transform = transforms.Compose([transforms.Grayscale(1),transforms.Resize(150),transforms.RandomHorizontalFlip(p=1),transforms.RandomVerticalFlip(p=0),transforms.ToTensor()])
    transformed_sample = img_transform(im_pil)

    im_pil.show()
    transforms.ToPILImage()(transformed_sample).show()

#test_transforms(img_test)
#test_transforms_mask(label_test)
#img_tests,label_tests = data_loader.get_image_and_labels(data_loader.valid_annotations)

#data_loader.plot_function(img_test,label_test)
def get_patches(images_idx,data_loader):
    img, mask = data_loader.get_image_and_labels(images_idx[0])
    images, masks = data_loader.generate_patches(img, mask)
    for i in images_idx[1:]:
        img_test, label_test = data_loader.get_image_and_labels(i)
        image, mask = data_loader.generate_patches(img_test, label_test)

        images = np.vstack((images, image))
        masks = np.vstack((masks, mask))
    return images,masks

def save_pictures_locally(data_loader,directory_path=r'C:\Users\Mads-\Documents\Universitet\5. Semester\Bachelorprojekt\data_folder'):
    for i in data_loader.valid_annotations:
        img,mask = data_loader.get_image_and_labels(i)
        os.chdir(directory_path)
        img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # cv2 has BGR channels, and Pillow has RGB channels, so they are transformed here
        im_pil = Image.fromarray(img2)
        im_pil.save(str(i)+".jpg")
        os.chdir(r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/training_mask')
        _, binary = cv2.threshold(mask * 255, 225, 255, cv2.THRESH_BINARY_INV)
        mask_pil = Image.fromarray(binary)
        mask_pil.convert('RGB').save(str(i)+'_mask.png')

if __name__ == '__main__':
    dataloader = DataLoader()
    pass
    images, masks = dataloader.get_image_and_labels([41,45])
    dataloader.plot_function(images,masks)


    img, mask = get_patches(dataloader.valid_annotations[0:50],dataloader)
    pass
    #images, masks = get_patches(np.where(np.array(dataloader.visibility_score) == 3)[0],dataloader)
    #dataloader.plot_function(images,masks)
    # pass
    # print(dataloader.visibility_score[:50])
    # pass
    #data_loader=DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',metadata_path=r'samples/model_comparison.csv')
    #save_pictures_locally(data_loader,directory_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/training_img')