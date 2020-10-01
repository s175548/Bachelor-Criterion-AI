import math
import sys
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn
from PIL import Image
from object_detect.get_bboxes import get_bbox_mask
import object_detect.helper.utils as utils
from object_detect.helper.evaluator import get_iou2, get_map2

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

def get_samples(samples,ids,N,path_save,train=True):
    for (img, m, t, p), id in zip(samples, ids):
        for i in range(len(ids)):
            boxes = p[i]['boxes'].detach().numpy()
            bmask = get_bbox_mask(mask=m[i], bbox=boxes)
            # image = (img[i].detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
            # Image.fromarray(img[i].numpy().astype(np.uint8)).save(path_save+'\_{}_img'.format(id),format='png')
            if train == False:
                Image.fromarray(bmask.astype(np.uint8)).save(path_save + '\_val_{}_{}_prediction.png'.format(N, id),
                                                             format='PNG')
            else:
                Image.fromarray(bmask.astype(np.uint8)).save(path_save + '\_train_{}_{}_prediction.png'.format(N, id),
                                                             format='PNG')


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, loss_list,risk=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    i = 0
    path_save = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Predictions_FRCNN'
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
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        ids = [targets[i]['image_id'].numpy() for i in range(len(targets))]
        #num_boxes.append(np.mean([len(targets[i]['boxes']) for i in range(len(ids))]))
        #num_boxes_pred.append(np.mean([len(targets[i]['boxes']) for i in range(len(ids))]))
        if risk==True:
            if i == 0:
                samples = []
                model.eval()
                outputs = model(images)
                num_boxes.append(np.mean([len(targets[i]['boxes']) for i in range(len(ids))]))
                num_boxes_pred.append(np.mean([len(outputs[i]['boxes']) for i in range(len(ids))]))
                model.train()
                samples.append((images, masks, targets, outputs))
                #get_samples(samples,ids,N=epoch,path_save=path_save,train=True)
        i+=1

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return model, np.mean(np.array(loss_list)), np.mean(np.array(num_boxes_pred)), np.mean(np.array(num_boxes))

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
def evaluate(model, data_loader, device,N,risk=True,threshold=0.5):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    path_save = r'C:\Users\johan\iCloudDrive\DTU\KID\BA\Kode\Predictions_FRCNN'
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    num_boxes_val = []
    num_boxes_pred = []
    i = 0
    mAP = []
    for (image, labels, masks) in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in image)
        targets = list({k: v.to(device, dtype=torch.long) for k,v in t.items()} for t in labels)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        ids = [targets[i]['image_id'].numpy() for i in range(len(targets))]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        for j in range(len(ids)):
            iou, index, selected_iou = get_iou2(boxes=outputs[j]['boxes'], target=targets[j]['boxes'])
            df, AP = get_map2(outputs[j]['boxes'], targets[j]['boxes'], outputs[j]['scores'], iou_list=selected_iou, threshold=0.5)
            mAP.append(AP)
        samples = []
        num_boxes_val.append(np.mean([len(targets[i]['boxes']) for i in range(len(ids))]))
        num_boxes_pred.append(np.mean([len(outputs[i]['boxes']) for i in range(len(ids))]))
        samples.append((images, masks, targets, outputs))
        if risk==True:
            if i < 10:
                get_samples(samples,ids,N=N,path_save=path_save,train=False)
        i+=1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("mean Average Precision for epoch {}: ".format(N+1), np.mean(mAP))
    # accumulate predictions from all images
    torch.set_num_threads(n_threads)
    return np.mean(mAP),np.mean(np.array(num_boxes_pred)),np.mean(np.array(num_boxes_val))