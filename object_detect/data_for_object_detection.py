import json, cv2, torch, os,pandas as pd,numpy as np
from PIL import Image
from torchvision import datasets, transforms
from data_import.draw_contours import draw_contours2,make_contour
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, data_path=r'C:\Users\Mads-\Desktop\leather_patches',
                 metadata_path=r'.samples\model_comparison.csv'):
        self.data_path = data_path
        self.metadata_path = os.path.join(data_path, metadata_path)
        self.metadata_csv = self.get_metadata(self.metadata_path)
        self.valid_annotations = self.get_empty_segmentations()
        self.binary_class_dictionary = self.generate_binary_class_dictionary()

    def get_metadata(self, metadata_path):
        """
        Collect the metadata_csv file containing 7 datapoints:
                0: category; 1: path; 2: etag; 3: segmentation_path; 4: segmentation_etag; 5: model_segmentation_path; 6: model_segmentation_etag
                (All categories can for example be retrieved by self.metadata_csv[:,0])
        """
        metadata_csv = pd.read_csv(metadata_path, sep=";")
        metadata_csv.to_numpy()
        return metadata_csv.to_numpy()

    def generate_binary_class_dictionary(self):
        """
        All categories found in metadata_csv are turned into dictionary, such that that can get a binary output (0: good, 1: defect) by parsing the category to the dict
        self.binary_class_dictionary[ self.metadata_csv[0,0] ] will return the binary value of the first datapoint.
        """
        binary_dict = {}
        for ele in np.unique(self.metadata_csv[:, 0]):
            if "good" in ele.lower():
                binary_dict[ele] = 0
            else:
                binary_dict[ele] = 1
        return binary_dict

    def get_image_and_labels(self, images_idx):
        """ Input: List of indices of the images one wants
            output: list of images, and list of corresponding masks
        """
        images = []
        segmentation_masks = []
        if type(images_idx) == list:
            for image_idx in images_idx:
                images.append(cv2.imread(os.path.join(self.data_path, self.metadata_csv[image_idx, 1])))
                segmentation_masks.append(
                    self.read_segmentation_file(os.path.join(self.data_path, self.metadata_csv[image_idx, 3])))
            return (images, segmentation_masks)
        else:
            return cv2.imread(
                os.path.join(self.data_path, self.metadata_csv[images_idx, 1])), self.read_segmentation_file(
                os.path.join(self.data_path, self.metadata_csv[images_idx, 3]))
        return

    # #seg = json.loads(read_file(segmentation_path).decode("utf-8"))
    # seg = json.loads(read_file(segmentation_path))
    # segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})

    def read_segmentation_file(self, filename):
        """helping function, that opens annotation file, converts it to a cv2 contour and returns contour/segmentation
        """
        fh = open(filename, "r")
        try:
            file_content = fh.read()
            seg = json.loads(file_content)
            segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
            return segmentation
        finally:
            fh.close()

    def get_empty_segmentations(self):
        """
        Filtering function. It opens the meta csv file and returns a list of indices with valid annotation/segmentations (the indices of the 10 files without valid segmentation are not included
        """
        empty = []
        for i in range(len(self.metadata_csv)):
            file_path = os.path.join(self.data_path, self.metadata_csv[i, 3])
            with open(file_path) as file:
                content = file.read()
                seg = json.loads(content)
                # content = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
                if seg['annotations'] == list():
                    empty.append((i, self.metadata_csv[i, 3]))
        return [i for i in range(len(self.metadata_csv)) if i not in [anno[0] for anno in empty]]
        # return empty

    def plot_function(self, images, masks=None):
        """input: image_idx
            The function will plot image, and masks (if given)
        """
        if isinstance(images, np.ndarray):
            cv2.imshow('image', images)
            cv2.waitKey(0)
            if isinstance(masks, np.ndarray):
                cv2.imshow('image', masks)
                cv2.waitKey(0)
        else:
            for idx, image in enumerate(images):
                cv2.imshow('image', image)
                cv2.waitKey(0)
                if masks != None:
                    cv2.imshow('image', masks[idx])
                    cv2.waitKey(0)


def to_tensor_and_normalize(img):
    img2 = cv2.cvtColor(img,
                        cv2.COLOR_BGR2RGB)  # cv2 has BGR channels, and Pillow has RGB channels, so they are transformed here
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tranformation = transforms.Compose([transforms.ToTensor(), normalize])
    a = tranformation(img2)
    # b = transforms.ToPILImage()(a)
    # b.show()
    return a


def test_transforms(img):
    img2 = cv2.cvtColor(img,
                        cv2.COLOR_BGR2RGB)  # cv2 has BGR channels, and Pillow has RGB channels, so they are transformed here
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose(
        [transforms.Resize(150), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
         transforms.ToTensor(), normalize])
    # transforms.RandomCrop(224)
    im_pil = Image.fromarray(img2)
    transformed_sample = img_transform(im_pil)

    im_pil.show()
    transforms.ToPILImage()(transformed_sample).show()


def test_transforms_mask(mask):
    _, binary = cv2.threshold(mask * 255, 225, 255, cv2.THRESH_BINARY_INV)
    im_pil = Image.fromarray(binary)
    img_transform = transforms.Compose(
        [transforms.Grayscale(1), transforms.Resize(150), transforms.RandomHorizontalFlip(p=1),
         transforms.RandomVerticalFlip(p=0), transforms.ToTensor()])
    transformed_sample = img_transform(im_pil)

    im_pil.show()
    transforms.ToPILImage()(transformed_sample).show()


def save_pictures_locally(data_loader,
                          directory_path=r'C:\Users\Mads-\Documents\Universitet\5. Semester\Bachelorprojekt\data_folder'):
    os.chdir(directory_path)
    for i in data_loader.valid_annotations:
        img, mask = data_loader.get_image_and_labels(i)

        img2 = cv2.cvtColor(img,
                            cv2.COLOR_BGR2RGB)  # cv2 has BGR channels, and Pillow has RGB channels, so they are transformed here
        im_pil = Image.fromarray(img2)
        im_pil.save(str(i) + ".png")

def save_masks_locally(data_loader,
                          directory_path=r'C:\Users\Mads-\Documents\Universitet\5. Semester\Bachelorprojekt\data_folder'):
    os.chdir(directory_path)
    for i in data_loader.valid_annotations:
        _, mask = data_loader.get_image_and_labels(i)

        _, binary = cv2.threshold(mask * 255, 225, 255, cv2.THRESH_BINARY_INV)
        mask_pil = Image.fromarray(binary)
        mask_pil.convert('RGB').save(str(i) + '_mask.png')

def generate_patches(img, msk,patch_size=256,print_=False):
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
                    cv2.waitKey(0)

                images.append(image)
                masks.append(mask)
    return images,masks

if __name__ == '__main__':
    data_loader = DataLoader(data_path=r'C:\Users\johan\OneDrive\Skrivebord\leather_patches',metadata_path=r'.samples\model_comparison.csv')
    save_pictures_locally(data_loader,r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Train_img')
    save_masks_locally(data_loader,r'C:\Users\johan\OneDrive\Skrivebord\leather_patches\Train_mask')
    pass