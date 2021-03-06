import math
import sys, os
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn
from PIL import Image
from object_detect.get_bboxes import get_bbox_mask
import object_detect.helper.utils as utils
from object_detect.helper.evaluator import get_iou2, get_map2, iou_multi, get_class_iou, classifier_metric, do_nms, get_iou_targets, check_iou, mask_iou
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


def train_one_epoch(model, model_name, optim_name, lr, optimizer, layers, data_loader, device, epoch, print_freq, loss_list,save_folder,risk=True,HPC=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    i = 0
    path_save = save_folder
    num_boxes = []
    num_boxes_pred = []
    for (images, labels, masks) in metric_logger.log_every(data_loader, print_freq, header):
        images = list(img.to(device, dtype=torch.float32) for img in images)
        targets = list({k: v.to(device) for k,v in t.items()} for t in labels)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        loss_list.append(loss_value)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("Target was: ", targets)
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        i+=1

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return model, np.mean(np.array(loss_list)), np.mean(np.array(num_boxes_pred)), np.mean(np.array(num_boxes))


def evaluate(model, model_name, optim_name, lr, layers, data_loader, device,N,loss_list,save_folder,risk=True,HPC=True,multi=False,scale=True,threshold=0.3):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    model.eval()
    if HPC:
        path_save = save_folder
    else:
        path_save = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Predictions_FRCNN'
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    num_boxes_val = []
    num_boxes_pred = []
    i = 0
    mIoU = []
    mAP = []
    mAP2 = []
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
                if multi:
                    iou, label_list = iou_multi(boxes=outputs[j]['boxes'].cpu(), targets=targets[j]['boxes'].cpu(),
                                                        pred=outputs[j]['labels'].cpu(), labels=targets[j]['labels'].cpu())
                    df, AP, AP2, c = get_class_iou(iou_list=iou,label_list=label_list,scores=outputs[j]['scores'].cpu(),
                                                   target=targets[j]['boxes'].cpu(), labels=targets[j]['labels'].cpu(),
                                                   preds=outputs[j]['labels'].cpu(),threshold=threshold)
                    mAP.append(AP)
                    mAP2.append(AP2)
                    if N % 50 == 0:
                        df2,_,_ = get_map2(outputs[j]['boxes'], targets[j]['boxes'], outputs[j]['scores'],
                                           outputs[j]['labels'].cpu(), targets[j]['labels'].cpu(), iou_list=iou, threshold=threshold,
                                           print_state=True)
                else:

                    boxes = outputs[j]['boxes'].cpu()
                    scores = outputs[j]['scores'].cpu()
                    preds = outputs[j]['labels'].cpu()

                    new_boxes, new_scores, new_preds = do_nms(boxes, scores, preds, threshold=0.2)
                    if scale == True:
                        expand = 42
                    else:
                        expand = 21
                    iou_target, iou_pred = get_iou_targets(boxes=new_boxes, targets=targets[j]['boxes'].cpu(),
                                                           preds=new_preds,
                                                           labels=targets[j]['labels'].cpu(), image=images[j],
                                                           expand=expand)
                    acc_dict = classifier_metric(iou_target, iou_pred, new_scores, targets[j]['boxes'].cpu(),
                                                 targets[j]['labels'].cpu())

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

                    iou_target2, iou_pred2 = get_iou_targets(boxes=boxes, targets=targets[j]['boxes'].cpu(),
                                                             preds=preds,
                                                             labels=targets[j]['labels'].cpu(), image=images[j],
                                                             expand=expand)
                    acc_dict2 = classifier_metric(iou_target2, iou_pred2, scores, targets[j]['boxes'].cpu(),
                                                  targets[j]['labels'].cpu())

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
                    iou, _, _ = get_iou2(boxes=new_boxes, targets=targets[j]['boxes'].cpu(),
                                          pred=new_preds, labels=targets[j]['labels'].cpu())
                    df, _, AP = get_map2(new_boxes, targets[j]['boxes'], new_scores,
                                           new_preds, targets[j]['labels'].cpu(), iou_list=iou, threshold=threshold)
                    iou2, _, _ = get_iou2(boxes=outputs[j]['boxes'].cpu(),
                                                        targets=targets[j]['boxes'].cpu(),
                                                        pred=outputs[j]['labels'].cpu(),
                                                        labels=targets[j]['labels'].cpu())
                    df2, _, AP2 = get_map2(outputs[j]['boxes'], targets[j]['boxes'], outputs[j]['scores'],
                                         outputs[j]['labels'].cpu(), targets[j]['labels'].cpu(), iou_list=iou2,
                                         threshold=threshold)
                    mAP.append(AP)
                    mAP2.append(AP2)
                    IoU = mask_iou(boxes=new_boxes,mask=masks[j],targets=targets[j]['boxes'].cpu())
                    mIoU.append(IoU[1])

            samples = []
            num_boxes_val.append(np.mean([len(targets[i]['boxes']) for i in range(len(ids))]))
            num_boxes_pred.append(np.mean([len(outputs[i]['boxes']) for i in range(len(ids))]))
            samples.append((images, masks, targets, outputs))
            if N % 40 == 0:
                metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)


            if risk==True:
                model.train()
                images = list(img.to(device, dtype=torch.float32) for img in image)
                targets = list({k: v.to(device) for k, v in t.items()} for t in labels)

                loss_dict = model(images, targets)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()
                loss_list.append(loss_value)

                model.eval()
            i+=1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if HPC:
            if N % 40 == 0:
                print("Averaged stats:", metric_logger)
                print("mean Average Precision for epoch {} with nms: ".format(N), np.mean(mAP))
                print("mean Average Precision without nms {}: ".format(N), np.mean(mAP2))
                print("mean IoU with nms: ", np.mean(mIoU))
                print("TP: ", conf_matrix["true_positives"])
                print("FP: ", conf_matrix["false_positives"])
                print("TN: ", conf_matrix["true_negatives"])
                print("FN: ", conf_matrix["false_negatives"])
                print("Total number of defects: ", conf_matrix["total_num_defects"])
                print("Images with good leather: ", conf_matrix["good_leather"])
                print("Images with bad leather: ", conf_matrix["bad_leather"])
        else:
            print("Averaged stats:", metric_logger)
            print("mean Average Precision for epoch {} with nms: ".format(N), np.mean(mAP))
            print("mean Average Precision without nms {}: ".format(N), np.mean(mAP2))
            print("mean IoU with nms: ", np.mean(mIoU))
            print("TP: ", conf_matrix["true_positives"])
            print("FP: ", conf_matrix["false_positives"])
            print("TN: ", conf_matrix["true_negatives"])
            print("FN: ", conf_matrix["false_negatives"])
            print("Total number of defects: ", conf_matrix["total_num_defects"])
            print("Images with good leather: ", conf_matrix["good_leather"])
            print("Images with bad leather: ", conf_matrix["bad_leather"])

    return np.mean(mAP),np.mean(mAP2),np.mean(loss_list),np.mean(np.array(num_boxes_pred)),np.mean(np.array(num_boxes_val)), conf_matrix, conf_matrix2, np.nanmean(mIoU)