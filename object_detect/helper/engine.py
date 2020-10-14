import math
import sys, os
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn
from PIL import Image
from object_detect.get_bboxes import get_bbox_mask
import object_detect.helper.utils as utils
from object_detect.helper.evaluator import get_iou2, get_map2
import matplotlib.pyplot as plt

def get_samples(samples,model_name,optim_name,lr,ids,N,path_save,train=True):
    for (img, m, t, p), id in zip(samples, ids):
        for i in range(len(ids)):
            boxes = p[i]['boxes'].detach().cpu().numpy()
            targets = t[i]['boxes'].detach().cpu().numpy()
            bmask = get_bbox_mask(mask=m[i], bbox=boxes)
            bmask2 = get_bbox_mask(mask=m[i], bbox=targets)
            if train == False:
                Image.fromarray(bmask.astype(np.uint8)).save(path_save + '/{}_{}_{}_val_{}_{}_prediction.png'.format(model_name,optim_name,lr,N, id.data),
                                                             format='PNG')
                Image.fromarray(bmask2.astype(np.uint8)).save(path_save + '/{}_{}_{}_truth_val_{}_{}_prediction.png'.format(model_name,optim_name,lr,N, id.data),
                                                             format='PNG')
            else:
                Image.fromarray(bmask.astype(np.uint8)).save(path_save + '/{}_{}_{}_train_{}_{}_prediction.png'.format(model_name,optim_name,lr,N, id.data),
                                                             format='PNG')
                Image.fromarray(bmask2.astype(np.uint8)).save(
                    path_save + '/{}_{}_{}_truth_train_{}_{}_prediction.png'.format(model_name, optim_name, lr, N, id.data),
                    format='PNG')
            if N == 100:
                image = (img[i].detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
                Image.fromarray(image).save(path_save + '{}_{}_img.png'.format(lr,id), format='png')


def train_one_epoch(model, model_name, optim_name, lr, optimizer, data_loader, device, epoch, print_freq, loss_list,save_folder,risk=True,HPC=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    i = 0
    path_save = save_folder
    num_boxes = []
    num_boxes_pred = []
    for (images, labels, masks) in metric_logger.log_every(data_loader, print_freq, header):
        images = list(img.to(device, dtype=torch.float32) for img in images)
        targets = list({k: v.to(device, dtype=torch.long) for k,v in t.items()} for t in labels)

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

        if lr_scheduler is not None:
            lr_scheduler.step()

        ids = [targets[l]['image_id'].cpu() for l in range(len(targets))]

        if risk==True:
            if i < 5:
                if HPC:
                    if epoch % 25 == 0:
                        samples = []
                        model.eval()
                        outputs = model(images)
                        num_boxes.append(np.mean([len(targets[j]['boxes']) for j in range(len(ids))]))
                        num_boxes_pred.append(np.mean([len(outputs[k]['boxes']) for k in range(len(ids))]))
                        model.train()
                        samples.append((images, masks, targets, outputs))
                        get_samples(samples,model_name,optim_name,lr,ids,N=epoch,path_save=path_save,train=True)
                else:
                    samples = []
                    model.eval()
                    outputs = model(images)
                    num_boxes.append(np.mean([len(targets[j]['boxes']) for j in range(len(ids))]))
                    num_boxes_pred.append(np.mean([len(outputs[k]['boxes']) for k in range(len(ids))]))
                    model.train()
                    samples.append((images, masks, targets, outputs))
                    get_samples(samples, model_name, optim_name, lr, ids, N=epoch, path_save=path_save, train=True)
        i+=1

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return model, np.mean(np.array(loss_list)), np.mean(np.array(num_boxes_pred)), np.mean(np.array(num_boxes))


def evaluate(model, model_name, optim_name, lr, data_loader, device,N,loss_list,save_folder,risk=True,HPC=True,threshold=0.3):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    if N % 25 == 0:
        print("N_threads: ", n_threads)
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
                iou, index, selected_iou = get_iou2(boxes=outputs[j]['boxes'].cpu(), targets=targets[j]['boxes'].cpu(),
                                                    pred=outputs[j]['labels'].cpu(), labels=targets[j]['labels'].cpu())
                df, AP, AP2 = get_map2(outputs[j]['boxes'], targets[j]['boxes'], outputs[j]['scores'],
                                       outputs[j]['labels'], targets[j]['labels'], iou_list=iou, threshold=threshold)
                mAP.append(AP)
                mAP2.append(AP2)
                if N % 50 == 0:
                    df2,_,_ = get_map2(outputs[j]['boxes'], targets[j]['boxes'], outputs[j]['scores'],
                                       outputs[j]['labels'], targets[j]['labels'], iou_list=iou, threshold=threshold,
                                       print_state=True)
                    #print(df2)
            samples = []
            num_boxes_val.append(np.mean([len(targets[i]['boxes']) for i in range(len(ids))]))
            num_boxes_pred.append(np.mean([len(outputs[i]['boxes']) for i in range(len(ids))]))
            samples.append((images, masks, targets, outputs))
            if N % 50 == 0:
                metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            if risk==True:
                model.train()
                loss_dict = model(images, targets)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()
                loss_list.append(loss_value)
                if i < 100:
                    if HPC:
                        if N % 100 == 0:
                            get_samples(samples,model_name,optim_name,lr,ids,N=N,path_save=path_save,train=False)
                    else:
                        get_samples(samples, model_name, optim_name, lr, ids, N=N, path_save=path_save, train=False)
                model.eval()
            i+=1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if HPC:
            if N % 25 == 0:
                print("Averaged stats:", metric_logger)
                print("mean Average Precision for epoch {}: ".format(N), np.mean(mAP))
                print("mean Average Precision with scores for epoch {}: ".format(N), np.mean(mAP2))
        else:
            print("Averaged stats:", metric_logger)
            print("mean Average Precision for epoch {}: ".format(N), np.mean(mAP))
            print("mean Average Precision with scores for epoch {}: ".format(N), np.mean(mAP2))
    return np.mean(mAP),np.mean(mAP2),np.mean(loss_list),np.mean(np.array(num_boxes_pred)),np.mean(np.array(num_boxes_val))