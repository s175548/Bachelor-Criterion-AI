import math
import sys
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn
from PIL import Image
from object_detect.get_bboxes import get_bbox_mask
from object_detect.helper.coco_utils import get_coco_api_from_dataset
from object_detect.helper.coco_eval import CocoEvaluator
import object_detect.helper.utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    for images, labels, _ in metric_logger.log_every(data_loader, print_freq, header):
        images = list(img.to(device, dtype=torch.float32) for img in images)
        targets = list({k: v.to(device, dtype=torch.long) for k,v in t.items()} for t in labels)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device,N):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    path_save = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Predictions_FRCNN'
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    num_boxes_val = []
    num_boxes_pred = []
    for (image, labels, masks) in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = list({k: v.to(device, dtype=torch.long) for k,v in t.items()} for t in labels)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        ids = [targets[i]['image_id'].numpy() for i in range(len(targets))]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        samples = []
        num_boxes_val.append([len(targets[i]['boxes']) for i in range(len(ids))])
        num_boxes_pred.append([len(targets[i]['boxes']) for i in range(len(ids))])
        samples.append((image, masks, targets, outputs))

        for (img,m,t,p), id in zip(samples,ids):
            for i in range(len(ids)):
                boxes = p[i]['boxes'].numpy()
                bmask = get_bbox_mask(mask=m[i], bbox=boxes)
                #image = (img[i].detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
                #Image.fromarray(img[i].numpy().astype(np.uint8)).save(path_save+'\_{}_img'.format(id),format='png')
                Image.fromarray(bmask.astype(np.uint8)).save(path_save+'\{}_{}_prediction.png'.format(N,id),format='PNG')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator