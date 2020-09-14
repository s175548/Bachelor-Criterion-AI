import numpy as np
import pandas as pd
#from skimage.data import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from data_import.data_loader import DataLoader


def convert_mask_to_bounding_box(mask):
    """input: mask
    output: bounding boxes
    """
    contours, hierarchy = cv2.findContours(mask.astype('uint8'), 1, 2)
    bounding_box_mask = np.empty((mask.shape[0],mask.shape[1]))
    bounding_box_coordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_box_mask = cv2.rectangle(bounding_box_mask.copy(), (x, y), (x + w, y + h), (255, 255, 255), 3)
        bounding_box_coordinates.append((x,y,w,h,1))
    return bounding_box_mask,bounding_box_coordinates

if __name__ == '__main__':
    data_loader = DataLoader()
    img_test, mask = data_loader.get_image_and_labels(1)

    test_mask,test_box_coord = convert_mask_to_bounding_box(mask)
    cv2.imshow('',test_mask)
    cv2.waitKey(0)
