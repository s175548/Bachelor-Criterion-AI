import math
import sys, os
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn
from PIL import Image
from object_detect.get_bboxes import get_bbox_mask
import object_detect.helper.utils as utils
from object_detect.helper.evaluator import get_non_maximum_supression, get_iou_targets, get_map2, classifier_metric
import matplotlib.pyplot as plt

def get_samples(samples,model_name,optim_name,lr,layers,ids,N,path_save,train=True):
    for (img, m, t, p), id in zip(samples, ids):
        for i in range(len(ids)):
            boxes = p[i]['boxes'].detach().cpu().numpy()
            targets = t[i]['boxes'].detach().cpu().numpy()
            bmask = get_bbox_mask(mask=m[i], bbox=boxes)
            bmask2 = get_bbox_mask(mask=m[i], bbox=targets)
            if train == False:
                Image.fromarray(bmask.astype(np.uint8)).save(path_save + '/{}_{}_{}_{}_val_{}_{}_prediction.png'.format(model_name,layers,optim_name,lr,N, id.data),
                                                             format='PNG')
                Image.fromarray(bmask2.astype(np.uint8)).save(path_save + '/{}_{}_{}_{}_truth_val_{}_{}_prediction.png'.format(model_name,layers,optim_name,lr,N, id.data),
                                                             format='PNG')
            else:
                Image.fromarray(bmask.astype(np.uint8)).save(path_save + '/{}_{}_{}_{}_train_{}_{}_prediction.png'.format(model_name,layers,optim_name,lr,N, id.data),
                                                             format='PNG')
                Image.fromarray(bmask2.astype(np.uint8)).save(
                    path_save + '/{}_{}_{}_{}_truth_train_{}_{}_prediction.png'.format(model_name, layers, optim_name, lr, N, id.data),
                    format='PNG')
            if N == 100:
                image = (img[i].detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
                Image.fromarray(image).save(path_save + '{}_{}_img.png'.format(lr,id), format='png')


def validate(model, model_name, data_loader, device, val=True, threshold=0.3):
    if val==True:
        path_save = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Predictions\binary\val'
    else:
        path_save = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\HPC\Predictions\binary\train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    i = 0
    total_num_defects = 0
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    mAP = []
    mAP2 = []
    with torch.no_grad():
        for (image, labels, masks) in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in image)
            targets = list({k: v.to(device, dtype=torch.long) for k,v in t.items()} for t in labels)

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            ids = [targets[i]['image_id'].cpu() for i in range(len(targets))]

            evaluator_time = time.time()
            evaluator_time = time.time() - evaluator_time

            for j in range(len(ids)):
                boxes = outputs[j]['boxes'].cpu()
                scores = outputs[j]['scores'].cpu()

                new_boxes, new_scores = get_non_maximum_supression(boxes, scores, iou_threshold=0.2)
                iou_target, iou_pred = get_iou_targets(boxes=new_boxes, targets=targets[j]['boxes'].cpu(), image=images[j])

                acc_dict = classifier_metric(iou_target, iou_pred, new_scores, targets[j]['boxes'].cpu())
                true_positives += acc_dict["Detected"]
                false_negatives += acc_dict["Defects"] - acc_dict["Detected"]
                false_positives += acc_dict["FP"]
                total_num_defects += acc_dict["Defects"]

                iou, index, selected_iou = get_iou_targets(boxes=boxes, targets=targets[j]['boxes'].cpu(),
                                                    pred=outputs[j]['labels'].cpu(), labels=targets[j]['labels'].cpu())

                _, AP, AP2 = get_map2(boxes, targets[j]['boxes'], scores,
                                       outputs[j]['labels'].cpu(), targets[j]['labels'].cpu(), iou_list=iou, threshold=0.3)
                mAP.append(AP)
                mAP2.append(AP2)

            samples = []
            samples.append((images, masks, targets, outputs))
            if N % 10 == 0:
                metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
            i+=1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
    return np.mean(mAP),np.mean(mAP2),np.mean(loss_list),np.mean(np.array(num_boxes_pred)),np.mean(np.array(num_boxes_val))