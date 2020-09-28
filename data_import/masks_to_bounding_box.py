import numpy as np,os
import pandas as pd
#from skimage.data import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from data_import.data_loader import DataLoader
from data_import.draw_contours import draw_contours2


def convert_mask_to_bounding_box(mask):
    """input: mask
    output: bounding boxes
    """
    contours, hierarchy = cv2.findContours(mask.astype('uint8'), 1, 2)
    bounding_box_mask = np.empty((mask.shape[0],mask.shape[1]))
    bounding_box_coordinates = []
    cv2.imshow('mask',mask)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_box_mask = cv2.rectangle(bounding_box_mask.copy(), (x, y), (x + w, y + h), (255, 255, 255), 3)
        bounding_box_coordinates.append((x,y,w,h))
    #cv2.imshow('bound', cv2.resize(bounding_box_mask,(1000,1000)))
    #cv2.waitKey(0)
    cv2.imshow('box',bounding_box_mask)
    cv2.waitKey(0)

    return bounding_box_mask,bounding_box_coordinates

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


def find_backgrounds(list):
    for i in list:
        img_test, mask = data_loader.get_image_and_labels(i)
        get_background_mask(img_test)

def combine_seg_and_back_mask(mask_idx):
    for i in mask_idx:
        img_, seg_mask = data_loader.get_image_and_labels(i)
        back_mask = get_background_mask(img_)
        new_mask =  np.squeeze(seg_mask) + back_mask*2
        #_, binary = cv2.threshold(new_mask * 255, 225, 255, cv2.THRESH_BINARY_INV)
        # mask_pil = Image.fromarray(new_mask)
        # mask_pil.convert('RGB').save(str(i)+'_mask.png')
        pass
    return new_mask

if __name__ == '__main__':
    data_loader = DataLoader()
    img_test, mask = data_loader.get_image_and_labels(1)
    background_idx = [41,42,56,99,102,121,153,157]
    bounding_boxes = []
    for idx in background_idx:
        _,mask = data_loader.get_image_and_labels(idx)
        _,bounding_box = convert_mask_to_bounding_box(mask)
        bounding_boxes.append(bounding_box)
    pass
    # find_background(img_test)
    find_backgrounds(background_idx)

    test_mask,test_box_coord = convert_mask_to_bounding_box(mask)
    pass
    #cv2.imshow('',test_mask)
    #cv2.waitKey(0)
