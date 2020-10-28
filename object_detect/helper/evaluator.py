""" Script for computing IoU for evaluation of object detection models"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
import object_detect.helper.utils as utils
import torchvision
from PIL import Image



def check_empty(scores,target,labels):
    if len(scores) == 0:
        if len(target) == 0:
            mAP = 1
            mAP2 = 1
            df = pd.DataFrame()
            print("Detected None, target None! :-) ")
        else:
            if len(labels) == 1:
                if labels == torch.zeros(1):
                    mAP = 1
                    mAP2 = 1
                    df = pd.DataFrame()
                    print("Detected None, target None! :-) ")
                else:
                    mAP = 0
                    mAP2 = 0
                    df = pd.DataFrame()
                    print("Detected None, target true :-( ")
            else:
                mAP = 0
                mAP2 = 0
                df = pd.DataFrame()
                print("Detected None, target true :-( ")
    return df, mAP, mAP2

def classifier_metric(iou_list,iou_pred,scores,target):
    acc_dict = {}
    if len(target) == 0:
        acc_dict["Defects"] = 0
        acc_dict["Detected"] = 0
        acc_dict["FP"] = len(scores)
    else:
        num_obj = len(target)
        true_labels = iou_list > 0
        counter = 0
        for i in range(num_obj):
            if true_labels[i] == True:
                counter += 1
            else:
                pass
        acc_dict["Defects"] = num_obj
        acc_dict["Detected"] = counter
        fp_count = [fp for fp in iou_pred if fp == 0]
        acc_dict["FP"] = len(fp_count)
    return acc_dict

def get_class_iou(iou_list,label_list,scores,target,labels,preds,threshold=0.3,print_state=False):
    c1, c2, c3 = [], [], []
    index = 0
    for l in label_list:
        if l == 1:
            c1.append(iou_list[index])
        elif l == 2:
            c2.append(iou_list[index])
        elif l == 3:
            c3.append(iou_list[index])
        index += 1
    c = np.array([np.mean(c1),np.mean(c2),np.mean(c3)])
    if len(scores) == 0:
        df, mAP, mAP2 = check_empty(scores,target,labels)
        return df, mAP, mAP2, c
    else:
        df = pd.DataFrame(scores.cpu().data,columns=["Scores"])
        true_labels = [iou_list >= threshold]
        df.insert(1,"Correct?",true_labels[0],True)
        df.insert(2,"IoU {}".format(threshold),iou_list,True)
        prec, rec = precision_recall(true_labels[0])
        df.insert(2,"Precision", prec,True)
        df.insert(3,"Recall", rec,True)
        mAP = average_precision_score(true_labels[0],preds.cpu())
        if len(scores) == 0:
            scores2 = np.zeros(len(true_labels[0]))
            mAP2 = average_precision_score(true_labels[0],scores2)
        else:
            mAP2 = average_precision_score(true_labels[0], scores.cpu())
        if np.isnan(mAP)==True:
            mAP = 0
        if np.isnan(mAP2) == True:
            mAP2 = 0
        if print_state==True:
            print("boxes: ", boxes)
            print("targets: ", target)
            print("iou: ", iou_list)
            print("scores: ", scores)
            print("predictions: ", preds)
            print("labels: ", labels)
            print("class iou: ", c)
    return df, mAP, mAP2, c

def iou_multi(boxes, targets, pred, labels):
    iou_list = np.array([])
    i = 0
    index_list = []
    iou_label_index = []
    for bbox in boxes:
        best_iou = 0
        xmin, ymin, xmax, ymax = bbox.unbind(0)
        bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)
        index = 0
        best_index = 0
        for target in targets:

            x1, y1, x2, y2 = target.unbind(0)
            target_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            xA = max(xmin, x1)
            yA = max(ymin, y1)
            xB = min(xmax, x2)
            yB = min(ymax, y2)

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(bbox_area + target_area - interArea)
            if iou > best_iou:
                if labels[index] == pred[i]:
                    best_iou = iou
                    best_index = index
            index += 1
        index_list.append(best_index)
        iou_label_index.append(labels[best_index])
        iou_list = np.append(iou_list, best_iou)
        i += 1
    new_iou_list = np.copy(iou_list)
    for j in index_list:
        all_preds = np.where(np.array(index_list) == j)
        best_pred = np.argmax(iou_list[all_preds])
        for io in all_preds[0]:
            if io == all_preds[0][best_pred]:
                pass
            else:
                new_iou_list[io] = 0
    if len(new_iou_list) == 0:
        new_iou_list = np.append(new_iou_list, 0)
        iou_list = np.append(iou_list, 0)
    return iou_list, iou_label_index

def get_non_maximum_supression(boxes,scores,iou_threshold):
    new_boxes = []
    new_scores = []
    if len(boxes) > 0:
        nms = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)
        for i in range(len(nms)):
            new_boxes.append(boxes[nms[i]])
            new_scores.append(scores[nms[i]])
        return new_boxes, new_scores
    else:
        return torch.tensor(new_boxes), torch.tensor(new_scores)

def get_iou_targets(boxes,targets,image,expand=256):
    iou_list = np.array([])
    iou_pred = np.zeros((len(boxes)))
    for target in targets:
        best_iou = 0
        xmin, ymin, xmax, ymax = target.unbind(0)
        if xmin >= expand:
            xmin -= expand
        else:
            xmin = torch.tensor(0)
        if ymin >= expand:
            ymin -= expand
        else:
            ymin = torch.tensor(0)
        if xmax <= image.shape[2]-expand:
            xmax += expand
        else:
            xmax = torch.tensor(image.shape[2])
        if ymax <= image.shape[1]-expand:
            ymax += expand
        else:
            ymax = torch.tensor(image.shape[1])
        target_area = (xmax - xmin + 1) * (ymax - ymin + 1)
        i = 0
        for bbox in boxes:

            x1, y1, x2, y2 = bbox.unbind(0)
            bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            xA = max(xmin, x1)
            yA = max(ymin, y1)
            xB = min(xmax, x2)
            yB = min(ymax, y2)

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(bbox_area + target_area - interArea)
            if iou > best_iou:
                best_iou = iou
                if best_iou > iou_pred[i]:
                    iou_pred[i] = best_iou
            i+=1
        iou_list = np.append(iou_list, best_iou)

    return iou_list, iou_pred

def get_iou2(boxes,targets, pred, labels):
    iou_list = np.array([])
    i = 0
    index_list = []
    iou_label_index = []
    for bbox in boxes:
        best_iou = 0
        xmin, ymin, xmax, ymax = bbox.unbind(0)
        bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)

        index = 0
        best_index = 0
        for target in targets:

            x1, y1, x2, y2 = target.unbind(0)
            target_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            xA = max(xmin, x1)
            yA = max(ymin, y1)
            xB = min(xmax, x2)
            yB = min(ymax, y2)

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(bbox_area + target_area - interArea)
            if iou > best_iou:
                if labels[index] == pred[i]:
                    best_iou = iou
                    best_index = index
            index +=1
        index_list.append(best_index)
        iou_label_index.append(labels[best_index])
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

def get_map2(boxes,target,scores,pred,labels,iou_list,threshold=0.3,print_state=False):
    map = []
    map2 = []
    if len(scores) == 0:
        df, mAP, mAP2 = check_empty(scores,target,labels)
        return df, mAP, mAP2
    else:
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
        else:
            mAP2 = average_precision_score(true_labels[0], scores.cpu())
        if np.isnan(mAP)==True:
            mAP = 0
        if np.isnan(mAP2) == True:
            mAP2 = 0
        if print_state==True:
            print("boxes: ", boxes)
            print("targets: ", target)
            print("iou: ", iou_list)
            print("scores: ", scores)
            print("predictions: ", pred)
            print("labels: ", labels)
        #if len(boxes) > 0:
        #print("boxes: ", boxes)
        #print("targets: ", target)
        #print("iou: ", iou_list)
        #print("scores: ", scores)
    return df, mAP, mAP2

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

def try_error():
    boxes = torch.tensor([[160.7921, 248.3389, 173.2974, 260.0414],
                          [161.3609, 247.1133, 173.5587, 254.6450],
                          [158.1664, 246.5395, 178.3249, 262.9370],
                          [158.8094, 249.3433, 166.4100, 259.2597],
                          [164.4904, 245.0411, 176.3329, 252.5500]], dtype=torch.float32)
    boxes2 = torch.tensor([[16, 248, 73, 260],
                           [161, 24, 173, 200],
                           [158, 210, 178, 262],
                           [158, 249, 166, 259],
                           [80, 90, 110, 120]], dtype=torch.float32)
    target = torch.tensor([[162, 248, 176, 261]], dtype=torch.float32)
    target2 = torch.tensor([[162, 248, 176, 261], [20, 232, 50, 255], [150, 30, 165, 210]], dtype=torch.float32)
    scores = torch.tensor([0.7861, 0.7633, 0.6983, 0.3056, 0.27], dtype=torch.float32)
    labels = torch.tensor([1, 2, 3], dtype=torch.int64)
    preds = torch.tensor([2, 3, 1, 1, 1], dtype=torch.int64)
    iou, label_list = iou_multi(boxes2, target2, preds, labels)
    # df, mAP, mAP2, c = get_class_iou(iou,label_list,scores,preds,threshold=0.1,print_state=True)
    # iou, index, selected_iou = get_iou2(boxes=boxes,target=target)
    # best_bboxes = boxes[index]
    # df, mAP, mAP2 = get_map2(boxes,target,scores,iou_list=iou,threshold=0.3,print_state=True)
    # print("IoU is; ", iou)
    # print("mAP is: ", c)

    acimg = []
    acdef = []
    ac1 = 0
    ac2 = 0

    for i in range(10):
        scores = np.array([1 / (2 + i), 1 / (3 + i), 1 / (4 + i)])
        if i % 2 == 0:
            target = []
            iou_list = np.zeros(3)
        else:
            target = np.ones(7)
            iou_list = np.array([1 / (1 + i), 1 / (2 + i), 1 / (3 + i)])
        acc_image, acc_def, acc1, acc2 = classifier_metric(iou_list, scores, target)
        acimg.append(acc_image)
        acdef.append(acc_def)
        ac1 += acc1
        ac2 += acc2

if __name__ == '__main__':
        from object_detect.load_data import train_loader
        device = torch.device('cpu')
        epoch = 1
        data_loader = train_loader
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        i = 0
        total_num_defects = 0
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        for (images, labels, masks) in metric_logger.log_every(data_loader, 1, header):
            images = list(img.to(device, dtype=torch.float32) for img in images)
            targets = list({k: v.to(device, dtype=torch.long) for k, v in t.items()} for t in labels)

            outputs = torch.tensor([[950, 0, 1500, 320],
                                  [1, 1, 1200, 1090],
                                  [500, 1200, 900, 1500],
                                    [550, 1250, 1050, 1400],
                                    [1000,10,1450,370],
                                    [10,10, 1400, 1200]], dtype=torch.float32)
            outputs2 = torch.tensor([[1, 1, 800, 1090]], dtype=torch.float32)
            scores2 = torch.tensor([0.7861], dtype=torch.float32)
            labels2 = torch.tensor([1], dtype=torch.int64)
            scores = torch.tensor([0.7861, 0.7633, 0.6983, 0.45, 0.35, 0.33], dtype=torch.float32)
            new_boxes, new_scores = get_non_maximum_supression(outputs,scores,iou_threshold=0.2)
            labels = torch.tensor([1, 1, 1], dtype=torch.int64)

            iou, iou_pred = get_iou_targets(boxes=new_boxes, targets=targets[9]['boxes'].cpu(),image=images[9])

           # df, AP, AP2 = get_map2(outputs2, targets[9]['boxes'], scores2,
           #                                labels2, targets[9]['labels'].cpu(), iou_list=iou, threshold=0.3)

            acc_dict = classifier_metric(iou, iou_pred, new_scores, targets[9]['boxes'].cpu())
            true_positives += acc_dict["Detected"]
            false_negatives += acc_dict["Defects"]-acc_dict["Detected"]
            false_positives += acc_dict["FP"]
            total_num_defects += acc_dict["Defects"]
            jo = 1
            #joh = torchvision.ops.nms(boxes, scores3, iou_threshold=0.2)
            break