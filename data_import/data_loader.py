import json, cv2, torch, os,pandas as pd,numpy as np,PIL
from PIL import Image
from torchvision import datasets, transforms
from data_import.draw_contours import draw_contours2,extract_bounding_box_coords
import matplotlib.pyplot as plt



class DataLoader():
    def __init__(self,data_path=r'C:\Users\Mads-\Desktop\leather_patches',metadata_path = r'samples\model_comparison.csv'):
        self.data_path = data_path
        self.metadata_path = os.path.join(data_path,metadata_path)
        self.metadata_csv = self.get_metadata(self.metadata_path)
        self.valid_annotations = self.get_empty_segmentations()
        self.annotations_dict=self.get_all_annotations()
        self.annotations_index=self.annotation_to_index()
        self.color_dict=self.make_color_dict()
        self.color_dict_binary={1:np.array([255,255,255]),self.annotations_dict["Background"]:self.color_dict[self.annotations_dict["Background"]]}


    def get_metadata(self,metadata_path):
        """     Collect the metadata_csv file containing 7 datapoints:
                0: category; 1: path; 2: etag; 3: segmentation_path; 4: segmentation_etag; 5: model_segmentation_path; 6: model_segmentation_etag
                (All categories can for example be retrieved by self.metadata_csv[:,0])
        """
        metadata_csv = pd.read_csv(metadata_path,sep=";")
        metadata_csv.to_numpy()
        return metadata_csv.to_numpy()

    def get_visibility_score(self):
        visibility_list = []
        for img_idx in range(len(self.metadata_csv)):
            filepath = os.path.join(self.data_path, self.metadata_csv[img_idx, 3][1:])
            ann = self.get_json_file_content(filepath)
            for a in ann["annotations"]:
                try:
                    visibility = a["visibility"]
                    visibility_list.append(img_idx)
                    break
                except KeyError:
                    pass
                if a["label"].startswith("visibility_"):
                    visibility = a["label"].split("_")[-1]
                    visibility_list.append(img_idx)
                    break
        return visibility_list




    def get_good_patches(self,save_path=None):
        images = []
        masks=[]

        for i in range(len(self.metadata_csv)):
            if self.metadata_csv[i, 0][2:6] == 'Good':
                image=np.array(PIL.Image.open(os.path.join(self.data_path, self.metadata_csv[i, 1])))
                mask=np.zeros(image.shape[:-1])
                images.append(image)
                masks.append(mask)
        return images, masks



    def get_image_and_labels(self,images_idx,labels="All",ignore_good=True,make_binary=True):
            """     input: give index/indices of the wanted images in the dataset
                    output: image(s) and mask(s) of the given index/indices
            """
            images = []
            segmentation_masks = []
            for image_idx in images_idx:
                image=np.array(PIL.Image.open(os.path.join(self.data_path, self.metadata_csv[image_idx,1])))
                mask=self.read_segmentation_file(os.path.join(self.data_path,self.metadata_csv[image_idx,3][1:]),ignore_good=ignore_good,make_binary=make_binary,labels=labels)
                back_mask=get_background_mask(image)
                mask=np.squeeze(mask)+np.array(back_mask)/255*self.annotations_dict["Background"]
                mask_3d=np.dstack((mask,mask,mask))
                if make_binary:
                    for color_id,color_map in self.color_dict_binary.items():
                        index = mask==color_id
                        mask_3d[index,:]=mask_3d[index,:]/color_id*color_map
                else:
                    for label in labels:
                        color_id=self.annotations_dict[label]
                        color_map=self.color_dict[color_id]
                        index = mask == color_id
                        mask_3d[index, :] = mask_3d[index, :] / color_id * color_map
                segmentation_masks.append(mask_3d.astype(np.uint8))
                images.append(image)
            return (images,segmentation_masks)


    def read_segmentation_file(self,filepath,ignore_good=True,make_binary=True,labels='All'):
        """     Helper function, that simply opens segmentation file, draws a contour from this.
                Output: Segmentation retrieved from filename
        """
        label_dict=self.annotations_dict.copy()
        seg = self.get_json_file_content(filepath)
        if ignore_good==True:
            key_good=[key for key in label_dict.keys() if key[:4]=="Good"]
        else:
            key_good=[]
        if make_binary==True:
            for key in label_dict.keys():
                label_dict[key]=1
        if labels=='All':
            labels=list(label_dict.keys())
        label_space = {kk["label"]: [label_dict[kk["label"]]] for kk in seg["annotations"] if
                       (kk["label"] in labels) and (kk["label"] not in key_good)}
        if not label_space:
            print('Image with provided idx does not contain any of the wanted labels')
            return
        else:
            segmentation = draw_contours2(seg, label_space=label_space)
            return segmentation

    def get_all_annotations(self):
        label_names_set=set()
        label_dict_new={}
        for annotation_path in self.metadata_csv[self.valid_annotations,3]:
            filepath=os.path.join(self.data_path,annotation_path[1:])
            seg = self.get_json_file_content(filepath)
            for label in seg["annotations"]:
                label_names_set.add(label["label"])
        for i,label_name in enumerate(np.sort(list(label_names_set))):
            label_dict_new[label_name]=i
        label_dict_new['Background']=len(list(label_dict_new.keys()))
        return label_dict_new

    def annotation_to_index(self):
        label_dict={key:set() for key in self.annotations_dict.keys()}
        for idx in self.valid_annotations:
            filepath = os.path.join(self.data_path, self.metadata_csv[idx,3][1:])
            seg = self.get_json_file_content(filepath)
            for label in seg["annotations"]:
                label_dict[label["label"]].add(idx)
        return label_dict

    def make_color_dict(self):
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(len(list(self.annotations_dict.keys()))+1)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        color_dict={}
        for color,label in zip(colors,np.sort(list(self.annotations_dict.values()))):
            color_dict[label]=color
        return color_dict

    def get_index_for_label(self,labels=None):
        label_idx=[]
        for label in labels:
            label_idx+=self.annotations_index[label]
        return label_idx

    def get_target_dict(self,labels="Binary"):
        if labels == "Binary":
            label_dict = {1:1,53:-1}
        else:
            label_dict={}
            for i,label in enumerate(labels):
                label_dict[self.annotations_dict[label]]=i+1
            label_dict['Background']=-1
        return label_dict



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
            file_path = os.path.join(self.data_path, self.metadata_csv[i, 3][1:])
            with open(file_path) as file:
                content = file.read()
                seg = json.loads(content)
                if seg['annotations'] == list():
                    empty.append((i,self.metadata_csv[i, 3][1:]))
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
    def generate_patches(self,img, msk,patch_size=512,print_=False,img_index=None):
        images = []
        masks = []
        crop_count_height = int(np.floor(img.shape[0] / patch_size))
        crop_count_width =  int(np.floor(img.shape[1] / patch_size))

        # seg = self.get_json_file_content( os.path.join(self.data_path, self.metadata_csv[img_index, 3]))
        # segmentation = extract_bounding_box_coords(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})

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
        else:
            images.append(img)
            masks.append(msk)
        return images,masks
    def enchance_contrast(self,img):
        img = Image.fromarray(img)
        img = PIL.ImageEnhance.Sharpness(img)
        img = img.enhance(2.0)
        img= PIL.ImageEnhance.Contrast(img)
        img=img.enhance(2.0)
        return np.array(img)

def convert_to_image(pred,color_dict,target_dict):
    rgb_pred = np.dstack((pred, pred, pred))
    for key, value in target_dict.items():
        print(key)
        color_id = key
        color_map = color_dict[key]
        print(color_map)
        index = pred == value
        print(sum(index))
        rgb_pred[index, :] = rgb_pred[index, :] / value * color_map
    return rgb_pred


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
        im_pil = Image.fromarray(img)
        im_pil.save(str(i)+".jpg")
        os.chdir(r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/training_mask')
        _, binary = cv2.threshold(mask * 255, 225, 255, cv2.THRESH_BINARY_INV)
        mask_pil = Image.fromarray(binary)
        mask_pil.convert('RGB').save(str(i)+'_mask.png')

def get_background_mask(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = mask1 + mask2
    median_mask = cv2.medianBlur(mask1, 5)
    return (~median_mask)




if __name__ == '__main__':
#    dataloader = DataLoader()
    data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',
                         metadata_path=r'samples/model_comparison.csv')
    label_dict=data_loader.get_image_and_labels([21],labels=['Piega ','Verruca','Puntura insetto'],make_binary=False)

#    images, masks = dataloader.get_image_and_labels(41)

#    dataloader.generate_patches(images,masks,img_index=41)

    # images, masks = dataloader.get_image_and_labels([41,45])
    # dataloader.plot_function(images,masks)


    # img, mask = get_patches(dataloader.valid_annotations[0:50],dataloader)

    #images, masks = get_patches(np.where(np.array(dataloader.visibility_score) == 3)[0],dataloader)
    #dataloader.plot_function(images,masks)
    # pass
    # print(dataloader.visibility_score[:50])
    # pass
    #save_pictures_locally(data_loader,directory_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/training_img')