import json, cv2, torch, os,pandas as pd,numpy as np
from torchvision import datasets, transforms
from data_import.draw_contours import draw_contours2,make_contour


class DataLoader():
    def __init__(self,data_path=r'C:\Users\Mads-\Desktop\leather_patches',metadata_path = r'.samples\model_comparison.csv'):
        self.data_path = data_path
        self.metadata_path = os.path.join(data_path,metadata_path)
        self.metadata_csv = self.get_metadata(self.metadata_path)
        self.binary_class_dictionary = self.generate_binary_class_dictionary()

    def get_metadata(self,metadata_path):
        """
        Collect the metadata_csv file containing 7 datapoints:
                0: category; 1: path; 2: etag; 3: segmentation_path; 4: segmentation_etag; 5: model_segmentation_path; 6: model_segmentation_etag
                (All categories can for example be retrieved by self.metadata_csv[:,0])
        """
        metadata_csv = pd.read_csv(metadata_path,sep=";")
        metadata_csv.to_numpy()
        return metadata_csv.to_numpy()

    def generate_binary_class_dictionary(self):
        """
        All categories found in metadata_csv are turned into dictionary, such that that can get a binary output (0: good, 1: defect) by parsing the category to the dict
        self.binary_class_dictionary[ self.metadata_csv[0,0] ] will return the binary value of the first datapoint.
        """
        binary_dict = {}
        for ele in np.unique(self.metadata_csv[:,0]):
            if "good" in ele.lower():
                binary_dict[ele] = 0
            else:
                binary_dict[ele] = 1
        return binary_dict

    def get_image_and_labels(self,images_idx):
        images = []
        segmentation_masks = []
        if type(images_idx) == list:
            for image_idx in images_idx:
                images.append( cv2.imread( os.path.join(self.data_path, self.metadata_csv[image_idx,1]) ))
                segmentation_masks.append( self.read_segmentation_file( os.path.join(self.data_path,self.metadata_csv[image_idx,3]) ) )
            return (images,segmentation_masks)
        else:
            return cv2.imread( os.path.join(self.data_path, self.metadata_csv[images_idx,1]) ),self.read_segmentation_file( os.path.join(self.data_path,self.metadata_csv[images_idx,3]))
        return

    # #seg = json.loads(read_file(segmentation_path).decode("utf-8"))
    # seg = json.loads(read_file(segmentation_path))
    # segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})

    def read_segmentation_file(self,filename):
        fh = open(filename, "r")
        try:
            file_content = fh.read()
            seg = json.loads(file_content)
            segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
            return segmentation
        finally:
            fh.close()


    def plot_function(self,images, masks = None):
        """input: image_idx
            The function will plot image, and masks (if given)
        """
        if isinstance(images,np.ndarray):
            cv2.imshow('image', images)
            cv2.waitKey(0)
            if isinstance(masks,np.ndarray):
                cv2.imshow('image', masks)
                cv2.waitKey(0)
        else:
            for idx,image in enumerate(images):
                cv2.imshow('image', image)
                cv2.waitKey(0)
                if masks != None:
                    cv2.imshow('image', masks[idx])
                    cv2.waitKey(0)

data_loader = DataLoader()
img_test,label_test = data_loader.get_image_and_labels(0)
img_tests,label_tests = data_loader.get_image_and_labels(list(range(3)))

#data_loader.plot_function(img_test,label_test)
data_loader.plot_function(img_tests, label_tests)

#
#     def normalize_data(self,path,data,batch_size = 256,num_workers = 4):
#         traindir = os.path.join(path, 'train')
#         valdir = os.path.join(path.data, 'val')
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#
#         train_loader = torch.utils.data.DataLoader(
#             datasets.ImageFolder(traindir, transforms.Compose([
#                 transforms.RandomSizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ])),
#             batch_size=batch_size, shuffle=True,
#             num_workers=num_workers, pin_memory=True)
#



#for name in image_names:
        #images.append(cv2.imread("./train_mini/"+name))
#train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
