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

def find_background(image):
    resized_img = cv2.resize(image,(512,512))
    cv2.imshow('image',resized_img)
    cv2.waitKey(0)

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2
    # cv2.imshow('mask1 and 2',mask1)
    # cv2.waitKey(0)

    test = mask1

    # blur = cv2.blur(test, (5, 5))
    # cv2.imshow('blur mask',blur)
    # cv2.waitKey(0)

    median = cv2.medianBlur(test, 5)
    resized_mask = cv2.resize(median,(512,512))

    cv2.imshow('median mask',resized_mask)
    cv2.waitKey(0)

    # blur1 = cv2.bilateralFilter(test, 9, 75, 75)
    # cv2.imshow('blur1 mask',blur1)
    # cv2.waitKey(0)


    # cv2.imshow('original',image)
    # cv2.imshow('hsv',hsv)
    # cv2.imshow('mask',mask)

    cv2.waitKey(0)

def test_find_backgrounds(list):
    for i in list:
        img_test, mask = data_loader.get_image_and_labels(i)
        find_background(img_test)


if __name__ == '__main__':
    data_loader = DataLoader()
    img_test, mask = data_loader.get_image_and_labels(1)
    background_idx = [1,39,41,42,56,99,102,121,153,157]
    # find_background(img_test)
    test_find_backgrounds(background_idx)

    #test_mask,test_box_coord = convert_mask_to_bounding_box(mask)
    #cv2.imshow('',test_mask)
    #cv2.waitKey(0)
