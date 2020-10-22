import numpy as np,os
import torch
import pandas as pd
#from skimage.data import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
#from data_import.data_loader import DataLoader
from data_import.draw_contours import draw_contours2
from PIL import Image
from scipy import ndimage

def create_boxes(masks,num_objs):
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin == xmax:
            if xmin < 254:
                xmax += 1
            else:
                xmin -= 1
        if ymin == ymax:
            if ymin < 254:
                ymax += 1
            else:
                ymin -= 1
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def check_mask(mask,name):
    j = np.shape(mask)
    np.save(name,j)
    return np.shape(mask)

def convert_mask_to_bbox(mask):
    """input: mask
    output: bounding boxes
    """
    new_mask = np.copy(mask)
    for i in range(np.shape(new_mask)[0]):
        for j in range(np.shape(new_mask)[0]):
            if mask[i,j] < 190:
                new_mask[i,j] = 0
            else:
                new_mask[i,j] = 1
    contours, hierarchy = cv2.findContours(new_mask.astype('uint8'), cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_L1)
    bounding_box_mask = np.empty((new_mask.shape[0],new_mask.shape[1]))
    bounding_box_coordinates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_box_mask = cv2.rectangle(bounding_box_mask.copy(), (x, y), (x + w, y + h), (255, 255, 255), 3)
        bounding_box_coordinates.append((x,y,x+w,y+h))

    return bounding_box_mask,bounding_box_coordinates

def transform_image(mask,mask_new,label):
    for i in range(np.shape(mask_new)[0]):
        for j in range(np.shape(mask_new)[1]):
            if mask[i, j] == label:
                pass
            else:
                mask_new[i, j] = 0
    return mask_new

def get_multi_bboxes(mask):
    """input: mask
    output: bounding boxes
    """
    s = ndimage.generate_binary_structure(2,2)
    new_mask, num_features = ndimage.label(mask, structure=s)

    labels = np.unique(mask)
    obj_ids = np.unique(new_mask)

    # first id is the background, so remove it
    labels = labels[1:]
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = new_mask == obj_ids[:, None, None]

    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    num_labels = len(labels)

    boxes = create_boxes(masks,num_objs)

    bounding_box_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    for box in boxes:
        x1, y1, x2, y2 = box
        bounding_box_mask = cv2.rectangle(bounding_box_mask.copy(), (x1, y1), (x2, y2), (255, 255, 255), 3)

    bboxes_labels = []
    bboxes = []
    obj_per_label = []
    for l in range(len(labels)):
        mask_new = np.copy(mask)
        mask_new = transform_image(mask,mask_new,labels[l])
        nm, _ = ndimage.label(mask_new, structure=s)
        num_objects = np.unique(nm)
        num_objects = num_objects[1:]
        masks_new = nm == num_objects[:, None, None]
        bbox = create_boxes(masks_new,len(num_objects))
        #bboxes.append(bbox[0])
        for k in range(len(num_objects)):
            if labels[l] == 4:
                #bboxes_labels.append(3)
                pass
            else:
                bboxes.append(bbox[0])
                bboxes_labels.append(labels[l])
        obj_per_label.append(k+1)

    bounding_box_mask2 = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    #colours = colours[1:]

    nl = list(labels)
    count = [[x,nl.count(x)] for x in set(nl)]
    count2 = [[x,bboxes_labels.count(x)] for x in set(bboxes_labels)]
    if len(bboxes) > len(bboxes_labels):
        print("REACHED")
    clist = [55, 110, 165, 220]
    k = 0
    labels2 = [l for l in labels if l != 4]
    print("count2: ", count2)
    print("labels: ", labels)
    for i in range(len(labels2)):
        k2 = count2[i][1]
        colour = count2[i][0]-1
        for box in bboxes[k:k+k2]:
                x1, y1, x2, y2 = box
                bounding_box_mask2 = cv2.rectangle(bounding_box_mask2.copy(), (x1, y1), (x2, y2), (clist[colour], 255, 255), 3)
        k = k2
    return bounding_box_mask2, bboxes, bboxes_labels, bounding_box_mask

def new_convert(mask):
    """input: mask
    output: bounding boxes
    """
    new_mask = np.copy(mask)
    #for i in range(np.shape(new_mask)[0]):
    #    for j in range(np.shape(new_mask)[0]):
    #        if mask[i,j] < 190:
    #            new_mask[i,j] = 0
    #        else:
    #            new_mask[i,j] = 1
    labeled_array, num_features = ndimage.label(new_mask)
    s = ndimage.generate_binary_structure(2,2)
    mask, num_features2 = ndimage.label(new_mask, structure=s)

    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]

    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin == xmax:
            if xmin < 254:
                xmax += 1
            else:
                xmin -= 1
        if ymin == ymax:
            if ymin < 254:
                ymax += 1
            else:
                ymin -= 1
        boxes.append([xmin, ymin, xmax, ymax])

    bounding_box_mask = np.empty((new_mask.shape[0], new_mask.shape[1]))
    for box in boxes:
        x1, y1, x2, y2 = box
        bounding_box_mask = cv2.rectangle(bounding_box_mask.copy(), (x1, y1), (x2, y2), (255, 255, 255), 3)
    #boxes = torch.as_tensor(boxes, dtype=torch.float32)

    return bounding_box_mask, boxes

def get_bbox_mask(mask,bbox):
    new_mask = np.copy(mask)
    for i in range(np.shape(new_mask)[0]):
        for j in range(np.shape(new_mask)[0]):
            if mask[i,j] > 0:
               new_mask[i,j] = 255
    bounding_box_mask = np.copy(new_mask)
    for i in range(len(bbox)):
        bounding_box_mask = cv2.rectangle(bounding_box_mask.copy(), (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), (155, 255, 0), 2)
    return bounding_box_mask


def find_background(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = mask1 + mask2
    median_mask = cv2.medianBlur(mask1, 5)
    return median_mask


def find_backgrounds(list):
    for i in list:
        img_test, mask = data_loader.get_image_and_labels(i)
        find_background(img_test)


if __name__ == '__main__':
    #data_loader = DataLoader()
    #img_test, mask = data_loader.get_image_and_labels(1)
    #background_idx = [41,42,56,99,102,121,153,157]
    #bounding_boxes = []
    #for idx in background_idx:
    #    _,mask = data_loader.get_image_and_labels(idx)
    #    _,bounding_box = convert_mask_to_bounding_box(mask)
    #    bounding_boxes.append(bounding_box)
    #pass
    # find_background(img_test)
    #find_backgrounds(background_idx)

    test_mask,test_box_coord = convert_mask_to_bounding_box(mask)
    pass
    #cv2.imshow('',test_mask)
    #cv2.waitKey(0)
