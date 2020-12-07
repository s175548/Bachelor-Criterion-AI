import math
import sys, os
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn
from PIL import Image
from object_detect.get_bboxes import get_bbox_mask, fill_bbox, expand_targets
import object_detect.helper.utils as utils
from object_detect.helper.evaluator import check_iou, get_iou2, get_iou_targets, get_map2, classifier_metric, do_nms
import matplotlib.pyplot as plt
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize


def get_predictions(samples,model_name,ids,path_save,file_names,val=False):
    for (img, m, t, p), id in zip(samples, ids):
        for i in range(len(ids)):
            image = (img[i].detach().cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
            boxes = p[i]['boxes'].detach().cpu().numpy()
            boxes2 = p[i]['boxes']
            targets = t[i]['boxes'].detach().cpu().numpy()
            label = t[i]['labels'].detach().cpu().numpy()
            scores = p[i]['scores']
            preds = p[i]['labels']
            new_boxes, _, _ = do_nms(boxes2,scores,preds,threshold=0.2)
            bmask = get_bbox_mask(mask=m[i], bbox=boxes)
            if label[0] == 0:
                bmask2 = get_bbox_mask(mask=m[i], bbox=np.array([]))
            else:
                bmask2 = get_bbox_mask(mask=m[i], bbox=targets)
            bmask3 = get_bbox_mask(mask=m[i], bbox=new_boxes.detach().cpu().numpy())
            if val == True:
                Image.fromarray(image.astype(np.uint8)).save(
                    path_save + '/{}_val_{}_img.png'.format(model_name,file_names[ids[i].numpy()[0]]),format='PNG')
                Image.fromarray(bmask3.astype(np.uint8)).save(
                    path_save + '/{}_val_{}_prediction.png'.format(model_name, file_names[ids[i].numpy()[0]]), format='PNG')
                Image.fromarray(bmask2.astype(np.uint8)).save(
                    path_save + '/{}_val_{}_target.png'.format(model_name, file_names[ids[i].numpy()[0]]), format='PNG')
            else:
                Image.fromarray(image.astype(np.uint8)).save(
                    path_save + '/{}_train_{}_img.png'.format(model_name,ids[i].numpy()[0]),format='PNG')
                Image.fromarray(bmask3.astype(np.uint8)).save(
                    path_save + '/{}_train_{}_prediction.png'.format(model_name, ids[i].numpy()[0]), format='PNG')
                Image.fromarray(bmask2.astype(np.uint8)).save(
                    path_save + '/{}_train_{}_target.png'.format(model_name, ids[i].numpy()[0]), format='PNG')


def validate(model, model_name, data_loader, device, path_save, bbox_type, file_names, resize=True, val=True, bbox=False, threshold=0.3):
    if bbox == False:
        if val == True:
            path_save = os.path.join(path_save, 'val')
        else:
            path_save = os.path.join(path_save, 'train')
    else:
        path_save = os.path.join(path_save, bbox_type)
    i = 0
    if resize:
        expand = 42
    else:
        expand = 84
    conf_matrix = {}
    conf_matrix["true_positives"] = 0
    conf_matrix["false_positives"] = 0
    conf_matrix["true_negatives"] = 0
    conf_matrix["false_negatives"] = 0
    conf_matrix["total_num_defects"] = 0
    conf_matrix["good_leather"] = 0
    conf_matrix["bad_leather"] = 0
    conf_matrix2 = {}
    conf_matrix2["true_positives"] = 0
    conf_matrix2["false_positives"] = 0
    conf_matrix2["true_negatives"] = 0
    conf_matrix2["false_negatives"] = 0
    conf_matrix2["total_num_defects"] = 0
    conf_matrix2["good_leather"] = 0
    conf_matrix2["bad_leather"] = 0
    mAP = []
    mAP2 = []
    with torch.no_grad():
        for (image, labels, masks) in data_loader:
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
                preds = outputs[j]['labels'].cpu()

                #new_boxes, new_scores, new_labels = get_non_maximum_supression(boxes, scores, preds, iou_threshold=0.2)
                new_boxes, new_scores, new_preds = do_nms(boxes,scores,preds,threshold=0.2)
                iou_target, iou_pred = get_iou_targets(boxes=new_boxes, targets=targets[j]['boxes'].cpu(), preds=new_preds,
                                                       labels=targets[j]['labels'].cpu(), image=images[j], expand=expand)
                iou_target2, iou_pred2 = get_iou_targets(boxes=boxes, targets=targets[j]['boxes'].cpu(), preds=preds,
                                                       labels=targets[j]['labels'].cpu(), image=images[j], expand=expand)
                acc_dict = classifier_metric(iou_target, iou_pred, new_scores, targets[j]['boxes'].cpu(), targets[j]['labels'].cpu())

                conf_matrix["true_positives"] += acc_dict["Detected"]
                conf_matrix["false_negatives"] += acc_dict["Defects"] - acc_dict["Detected"]
                conf_matrix["false_positives"] += acc_dict["FP"]
                conf_matrix["total_num_defects"] += acc_dict["Defects"]
                if acc_dict["Defects"] == 0:
                    conf_matrix["good_leather"] += 1
                    if acc_dict["FP"] == 0:
                        conf_matrix["true_negatives"] += 1
                else:
                    conf_matrix["bad_leather"] += 1

                acc_dict2 = classifier_metric(iou_target2, iou_pred2, scores, targets[j]['boxes'].cpu(), targets[j]['labels'].cpu())

                conf_matrix2["true_positives"] += acc_dict2["Detected"]
                conf_matrix2["false_negatives"] += acc_dict2["Defects"] - acc_dict2["Detected"]
                conf_matrix2["false_positives"] += acc_dict2["FP"]
                conf_matrix2["total_num_defects"] += acc_dict2["Defects"]
                if acc_dict2["Defects"] == 0:
                    conf_matrix2["good_leather"] += 1
                    if acc_dict2["FP"] == 0:
                        conf_matrix2["true_negatives"] += 1
                else:
                    conf_matrix2["bad_leather"] += 1

                iou, index, selected_iou = get_iou2(boxes=outputs[j]['boxes'].cpu(), targets=targets[j]['boxes'].cpu(),
                                                    pred=outputs[j]['labels'].cpu(), labels=targets[j]['labels'].cpu())
                df, _, AP = get_map2(outputs[j]['boxes'], targets[j]['boxes'], outputs[j]['scores'],
                                     outputs[j]['labels'].cpu(), targets[j]['labels'].cpu(), iou_list=iou,
                                     threshold=threshold)
                iou2, _, _ = get_iou2(boxes=new_boxes, targets=targets[j]['boxes'].cpu(),
                                      pred=new_preds, labels=targets[j]['labels'].cpu())
                df2, _, AP2 = get_map2(new_boxes, targets[j]['boxes'], new_scores,
                                       new_preds, targets[j]['labels'].cpu(), iou_list=iou2, threshold=threshold)
                mAP.append(AP)
                mAP2.append(AP2)

            samples = []
            samples.append((images, expanded_targets, targets, outputs))
            get_predictions(samples, model_name, ids, path_save=path_save, file_names=file_names,val=val)
            i+=1

    return np.mean(mAP),np.mean(mAP2), conf_matrix, conf_matrix2
