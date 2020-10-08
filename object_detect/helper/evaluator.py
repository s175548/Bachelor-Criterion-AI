""" Script for computing IoU for evaluation of object detection models"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


def get_iou(boxes,target):
    iou_list = np.array([])
    i = 0
    index_list = []
    for label in target:
        best_iou = 0

        x1, y1, x2, y2 = label.unbind(0)
        target_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        index = 0
        best_index = 0
        for bbox in boxes:

            xmin, ymin, xmax, ymax = bbox.unbind(0)
            bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)

            xA = max(xmin, x1)
            yA = max(ymin, y1)
            xB = min(xmax, x2)
            yB = min(ymax, y2)

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(bbox_area + target_area - interArea)
            if iou > best_iou:
                best_iou = iou
                best_index = index

            index +=1
        index_list.append(best_index)
        iou_list = np.append(iou_list, best_iou)
        i+=1
    return iou_list, index_list

def get_iou2(boxes,target):
    iou_list = np.array([])
    i = 0
    index_list = []
    bbox_index = 0
    for bbox in boxes:
        best_iou = 0
        xmin, ymin, xmax, ymax = bbox.unbind(0)
        bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)

        index = 0
        best_index = 0
        for label in target:

            x1, y1, x2, y2 = label.unbind(0)
            target_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            xA = max(xmin, x1)
            yA = max(ymin, y1)
            xB = min(xmax, x2)
            yB = min(ymax, y2)

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(bbox_area + target_area - interArea)
            if iou > best_iou:
                best_iou = iou
                best_index = index

            index +=1
        index_list.append(best_index)
        iou_list = np.append(iou_list, best_iou)
        i+=1
    new_iou_list = np.copy(iou_list)
    for j in index_list:
        all_preds = np.where(np.array(index_list) == j)
        best_pred = np.argmax(iou_list[all_preds])
        for io in all_preds[0]:
            if io == all_preds[0][best_pred]:
                pass
            else:
                new_iou_list[io] = 0
    if len(new_iou_list)==0:
        new_iou_list = np.append(new_iou_list, 0)
        iou_list = np.append(iou_list, 0)
    return iou_list, index_list, new_iou_list

def get_map2(boxes,target,scores,iou_list,threshold=0.3,print_state=False):
    df = pd.DataFrame(scores.cpu().data,columns=["Scores"])
    true_labels = [iou_list >= threshold]
    df.insert(1,"Correct?",true_labels[0],True)
    df.insert(2,"IoU {}".format(threshold),iou_list,True)
    prec, rec = precision_recall(true_labels[0])
    pred = np.ones((len(true_labels[0])))
    df.insert(2,"Precision", prec,True)
    df.insert(3,"Recall", rec,True)
    mAP = average_precision_score(true_labels[0],pred)
    if len(scores) == 0:
        scores2 = np.zeros(len(true_labels[0]))
        mAP2 = average_precision_score(true_labels[0],scores2)
    for j in range(len(scores)):
        if torch.isnan(scores[j].detach().cpu()) == True:
            scores[j] = torch.tensor(0, dtype =torch.float32)
    else:
        mAP2 = average_precision_score(true_labels[0], scores.cpu())
    if np.isnan(mAP)==True:
        mAP = 0
    if print_state==True:
        print("boxes: ", boxes)
        print("targets: ", target)
        print("iou: ", iou_list)
        print("scores: ", scores)
    #if len(boxes) > 0:
        #print("boxes: ", boxes)
        #print("targets: ", target)
        #print("iou: ", iou_list)
        #print("scores: ", scores)
    return df, mAP, mAP2

def get_map(boxes,target,scores,iou_list,threshold=0.5):
    zipped = zip(scores.numpy(), boxes.numpy(), target.numpy())
    #sort_zip = sorted(zipped)
    sorted_zip = sorted(zipped, key=lambda x: x[0], reverse=True)
    df = pd.DataFrame(sorted_zip,columns=["Scores","Boxes","Target"])
    sc, bo, ta = [], [], []
    for i in range(len(sorted_zip)):
        sc.append(sorted_zip[i][0])
        bo.append(sorted_zip[i][1])
        ta.append(sorted_zip[i][2])
    sc, bo, ta = np.array(sc), np.array(bo), np.array(ta)
    true_labels = [iou_list >= threshold]
    df.insert(0,"Correct?",true_labels[0],True)
    df.insert(1,"IoU {}".format(threshold),iou_list,True)
    prec, rec = precision_recall(true_labels[0])
    pred = np.ones((len(true_labels[0])))
    df.insert(1,"Precision", prec,True)
    df.insert(2,"Recall", rec,True)
    mAP = average_precision_score(true_labels[0],pred)
    if np.isnan(mAP)==True:
        mAP = 0
    return df, mAP

def precision_recall(pred):
    precision, recall = np.zeros((len(pred))), np.zeros((len(pred)))
    running_pos = 0
    running_neg = 0
    for i in range(len(pred)):
        if pred[i] == True:
            running_pos += 1
            precision[i] = running_pos/(i+1)
            recall[i] = running_pos/len(pred)
        else:
            running_neg += 1
            precision[i] = 1-running_neg/(i+1)
            recall[i] = running_pos/len(pred)
    return precision, recall


if __name__ == '__main__':
    boxes = torch.tensor([[160.7921, 248.3389, 173.2974, 260.0414],
                          [161.3609, 247.1133, 173.5587, 254.6450],
                          [158.1664, 246.5395, 178.3249, 262.9370],
                          [158.8094, 249.3433, 166.4100, 259.2597],
                         [164.4904, 245.0411, 176.3329, 252.5500]],dtype=torch.float32)
    target = torch.tensor([[162, 248, 176, 261]],dtype=torch.float32)
    scores = torch.tensor([0.7861, 0.7633, 0.6983, 0.3056, float('NaN')],dtype=torch.float32)
    iou, index, selected_iou = get_iou2(boxes=boxes,target=target)
    #best_bboxes = boxes[index]
    df, mAP, mAP2 = get_map2(boxes,target,scores,iou_list=iou,threshold=0.3,print_state=True)
    print("IoU is; ", iou)
    print("mAP is: ", mAP)