import json, cv2, torch, os,pandas as pd,numpy as np,PIL
from PIL import Image
from torchvision import datasets, transforms
from data_import.draw_contours import draw_contours2,extract_bounding_box_coords
import matplotlib.pyplot as plt



class DataLoader():
    def __init__(self,data_path=r'C:\Users\Mads-\Desktop\leather_patches',metadata_path = r'samples/model_comparison.csv'):
        self.data_path = data_path
        self.metadata_path = os.path.join(data_path,metadata_path)
        self.insect_bite_names=['Area Punture insetti', 'Insect bite', "Puntura d'insetto", 'Puntura insetto']
        self.metadata_csv = self.get_metadata(self.metadata_path)
        self.valid_annotations = self.get_empty_segmentations()
        self.annotations_dict=self.get_all_annotations()
        self.annotations_index=self.annotation_to_index()
        self.color_dict=self.make_color_dict()
        self.color_dict_binary=self.make_color_dict(binary=True)


    def get_metadata(self,metadata_path):
        """     Collect the metadata_csv file containing 7 datapoints:
                0: category; 1: path; 2: etag; 3: segmentation_path; 4: segmentation_etag; 5: model_segmentation_path; 6: model_segmentation_etag
                (All categories can for example be retrieved by self.metadata_csv[:,0])
        """
        metadata_csv = pd.read_csv(metadata_path,sep=";")
        metadata_csv.to_numpy()
        return metadata_csv.to_numpy()

    def get_visibility_score(self,scores=[2,3]):
        visibility_list = []
        for img_idx in range(len(self.metadata_csv)):
            filepath = os.path.join(self.data_path, self.metadata_csv[img_idx, 3][1:])
            ann = self.get_json_file_content(filepath)
            for a in ann["annotations"]:
                try:
                    visibility = a["visibility"]
                    if int(visibility) in scores:
                        visibility_list.append(img_idx)
                    break
                except KeyError:
                    pass
                if a["label"].startswith("visibility_"):
                    visibility = a["label"].split("_")[-1]
                    if int(visibility) in scores:
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



    def get_image_and_labels(self,images_idx,labels="All",ignore_good=False,make_binary=True):
            """     input: give index/indices of the wanted images in the dataset
                    output: image(s) and mask(s) of the given index/indices
            """
            images = []
            segmentation_masks = []
            if labels=='All':
                labels=self.annotations_dict.keys()
            if make_binary:
                color_map_dict = self.color_dict_binary
            else:
                color_map_dict = self.color_dict
            for image_idx in images_idx:
                image=np.array(PIL.Image.open(os.path.join(self.data_path, self.metadata_csv[image_idx,1])))
                mask=self.read_segmentation_file(os.path.join(self.data_path,self.metadata_csv[image_idx,3][1:]),labels=labels)
                back_mask=get_background_mask(image)
                back_mask[(np.array(back_mask)!=0) & (np.squeeze(mask)!=0)]=0
                mask=np.squeeze(mask)+np.array(back_mask)/255*self.annotations_dict["Background"]
                mask_3d=np.dstack((mask,mask,mask))
                for label in labels:
                    color_id=self.annotations_dict[label]
                    color_map=color_map_dict[color_id]
                    index = mask == color_id
                    mask_3d[index, :] = mask_3d[index, :] / color_id * color_map
                segmentation_masks.append(mask_3d.astype(np.uint8))
                images.append(image)
            return (images,segmentation_masks)


    def read_segmentation_file(self,filepath,labels='All'):
        """     Helper function, that simply opens segmentation file, draws a contour from this.
                Output: Segmentation retrieved from filename
        """
        label_dict=self.annotations_dict.copy()
        seg = self.get_json_file_content(filepath)
        if labels=='All':
            labels=list(label_dict.keys())
        label_space = {kk["label"]: [label_dict[kk["label"]]] for kk in seg["annotations"] if
                       (kk["label"] in labels)}
        if not label_space:
            print('Image with provided idx does not contain any of the wanted labels')
            return
        else:
            segmentation = draw_contours2(seg, label_space=label_space)
            return segmentation

    def get_separate_segmentations(self,filepath,labels):
        seg = self.get_json_file_content(filepath)
        segmentations_to_return=[]
        label_space = {kk["label"]: [1] for kk in seg["annotations"] if kk["label"] in labels}
        for kk in seg['annotations']:
            if kk["label"] in labels:
                seg_dict=seg
                seg_dict['annotations']=[kk]
                segmentation = draw_contours2(seg_dict, label_space=label_space)
                if kk['label'] in self.insect_bite_names:
                    segmentations_to_return.append(('Insect bite',segmentation))
                else:
                    segmentations_to_return.append(('Binary', segmentation))
        return segmentations_to_return








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

    def get_idx_from_single_skin(self,skin='WALKNAPPA'):
        idx_list=[]
        for idx in self.valid_annotations:
            path=self.metadata_csv[idx,1]
            if path.lower()[:3]==skin.lower()[:3]:
                idx_list.append(idx)
        return idx_list

    def test_training_split(self,p_value=[0.65,0.01]):
        train_idx=[]
        val_idx=[]
        y_thresh=p_value[0]*65000
        p_value.reverse()
        idx_to_include=load_idx_to_include()
        idx_to_include=np.intersect1d(idx_to_include,self.valid_annotations)
        split=self.metadata_csv[0, 1].split('/')[2]
        for idx in idx_to_include[1:]:
            path = self.metadata_csv[idx, 1].split('/')
            if (path[0][0]=='W') & (split!=False) :
                y_thresh=65000*p_value[0]
                p_value.reverse()
                split=self.metadata_csv[idx, 1].split('/')[2]
                split=False
            img_size=path[-1].split('x')
            img_size_y=img_size[0].split('.')[0]
            if int(img_size_y)>y_thresh:
                if p_value[0]>0.5:
                    train_idx.append(idx)
                else:
                    val_idx.append(idx)
            else:
                if p_value[0]>0.5:
                    val_idx.append(idx)
                else:
                    train_idx.append(idx)
        return train_idx,val_idx

    def annotation_to_index(self,index_list=[]):
        if index_list==[]:
            index_list=self.valid_annotations
        label_dict={key:set() for key in self.annotations_dict.keys()}
        for idx in index_list:
            filepath = os.path.join(self.data_path, self.metadata_csv[idx,3][1:])
            seg = self.get_json_file_content(filepath)
            for label in seg["annotations"]:
                label_dict[label["label"]].add(idx)
        return label_dict

    def make_color_dict(self,binary=False):
        color_dict={}
        np.random.seed(0)
        colors = np.array([[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)] for _ in
                  range(60)])
        for color,key_val in zip(colors,sorted(list(self.annotations_dict.items()))):
            if key_val[0][:4] == "Good":
                color_dict[int(key_val[1])]=np.array([0,0,0])
#            elif binary and key_val[0] in self.insect_bite_names:
#                color_dict[int(key_val[1])] = colors[0]
            elif binary and key_val[0] != 'Background':
                color_dict[int(key_val[1])] = np.array([255, 255, 255])
            else:
                color_dict[int(key_val[1])]=color
        return color_dict

    def get_index_for_label(self,labels=None):
        label_idx=[]
        for label in labels:
            label_idx+=self.annotations_index[label]
        return label_idx

    def get_target_dict(self,labels="Binary"):
        if labels == "Binary":
            label_dict = {1:1,53:2}
        else:
            label_dict={}
            for i,label in enumerate(labels):
                label_dict[self.annotations_dict[label]]=i+1
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

    def pad_tif(self,image,extra_shape = 50):
        h,w,c = image.shape[0]+extra_shape,image.shape[1]+extra_shape,3
        extra_h= h - image.shape[0] % h
        extra_w = w- image.shape[1] % w
        padded_img = np.pad(image, (extra_h, extra_w), 'reflect')
        return padded_img

    def generate_tif_patches(self, img, patch_size=512,padding = 100,with_pad = False):
        crop_count_height = img.shape[0] // patch_size
        crop_count_width = img.shape[1] // patch_size
        n_imgs = crop_count_height * crop_count_width

        split_dimensions = (patch_size, patch_size, 3)
        split_imgs = np.empty((n_imgs, *split_dimensions))
        if with_pad:
            pad_split_dimensions = (img.shape[0]+padding, img.shape[1]+padding, 3)
            pad_split_imgs = np.empty((n_imgs, *pad_split_dimensions))
            padded_img = self.pad_tif(img, padding//2)
        else:
            pad_split_imgs = np.empty([0])


        for i in range(crop_count_height):
            for j in range(crop_count_width):
                image = img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                split_imgs[i*crop_count_width+j] = image
                if with_pad:
                    large_img = padded_img[i * patch_size:(i + 1) * patch_size+padding, j * patch_size:(j + 1) * patch_size+padding]
                    pad_split_imgs[i*crop_count_width+j] = large_img

        return split_imgs, (crop_count_height,crop_count_width ), pad_split_imgs

def load_idx_to_include():
    idx=open(os.path.join(os.getcwd(),'data_import','idx_to_include.txt'),'r')
    idx=idx.read()
    idx=idx.split(' ')
    while '' in idx:
        idx.remove('')
    idx=[int(id) for id in idx]
    return idx



def convert_to_image(pred,color_dict,target_dict):
    rgb_pred = np.dstack((pred, pred, pred))
    for key, value in target_dict.items():
        color_id = key
        color_map = color_dict[key]
        index = pred == value
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
    data_loader = DataLoader(data_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /leather_patches',
                             metadata_path=r'samples/model_comparison.csv')
    #train,val=data_loader.test_training_split()
    #for data in [train,val] :
    #    idx_dict=np.intersect1d(data,data_loader.get_visibility_score([2,3]))
    #    index=data_loader.annotation_to_index(index_list=idx_dict)


    #label_dict=data_loader.get_image_and_labels([21],labels=['Piega ','Verruca','Puntura insetto'],make_binary=False)

#    images, masks = dataloader.get_image_and_labels(41)

#    dataloader.generate_patches(images,masks,img_index=41)

    # images, masks = dataloader.get_image_and_labels([41,45])
    # dataloader.plot_function(images,masks)


    # img, mask = get_patches(dataloader.valid_annotations[0:50],dataloader)

    #images, masks = get_patches(np.where(np.array(dataloader.visibility_score) == 3)[0],dataloader)
    #dataloader.plot_function(images,masks)
    # pass
    #
# (dataloader.visibility_score[:50])
    # pass
    #save_pictures_locally(data_loader,directory_path=r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/training_img')