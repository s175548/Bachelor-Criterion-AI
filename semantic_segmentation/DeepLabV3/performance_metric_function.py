from data_import.data_loader import DataLoader
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import PIL
import torchvision.transforms.functional as F
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics



def error_count(idx, pred_color, target_color, data_loader, labels, errors, false_positives,true_negatives, metric, resize=False,
                size=None, scale=None,centercrop=False,path=None):
    pred = pred_color.copy()
    target = target_color.copy()
    if np.sum(target == 1) != 0:
        if path != None:
            masks = data_loader.get_separate_segmentations(
                path, labels=labels)
        else:
            masks = data_loader.get_separate_segmentations(
                os.path.join(data_loader.data_path, data_loader.metadata_csv[idx, 3][1:]), labels=labels,)
        buffer = 84
        xdim_s = []
        ydim_s = []
        for i,mask in enumerate(masks):
            print(i)
            label, mask = mask[0], np.squeeze(np.array(mask[1]).astype(np.uint8))
            if centercrop:
                if size > np.min(mask.shape):
                    pass
                else:
                    mask = F.center_crop(PIL.Image.fromarray(mask), output_size=size)
            mask = np.array(mask)
            if resize:
                resize_shape = (int(mask.shape[0] * scale), int(mask.shape[1] * scale))
                mask = F.resize(PIL.Image.fromarray(mask), resize_shape, PIL.Image.NEAREST)
            mask=np.array(mask).astype(np.uint8)
            if np.sum(mask == 1) != 0:
                row, col = np.where(mask != 0)
                xdim = (np.maximum(np.min(row) - buffer, 0), np.minimum(np.max(row) + buffer, mask.shape[0]))
                xdim_s.append(xdim)
                ydim = (np.maximum(np.min(col) - buffer, 0), np.minimum(np.max(col) + buffer, mask.shape[1]))
                ydim_s.append(ydim)
                mask = mask[xdim[0]:xdim[1], ydim[0]:ydim[1]]
                defect_found = int(np.sum(pred[xdim[0]:xdim[1], ydim[0]:ydim[1]] != 0) > 0)
                if label == 'Insect bite':
                    errors[1, 0] += 1
                    errors[1, 1] += defect_found
                    metric[0].update(mask, pred[xdim[0]:xdim[1], ydim[0]:ydim[1]])
                else:
                    errors[0, 0] += 1
                    errors[0, 1] += defect_found
                    metric[1].update(mask, pred[xdim[0]:xdim[1], ydim[0]:ydim[1]])
            else:
                pass
        for xdim, ydim in zip(xdim_s, ydim_s):
            pred[xdim[0]:xdim[1], ydim[0]:ydim[1]] = 0
            target[xdim[0]:xdim[1], ydim[0]:ydim[1]] = 0
    else:
        xdim_s = None
        ydim_s = None
        true_negatives[0]+=1
        if np.sum(pred == 1) == 0:
            true_negatives[1]+=1

    if np.sum(pred == 1) > 0:
        false_positives += 1
    target[target == 2] = 0
    metric[2].update(target, pred)
    target_color[target_color == 2] = 0
    target_color, pred_color = color_target_pred(target_color, pred_color, pred, xdim_s, ydim_s)
    return errors, false_positives, metric, target_color, pred_color,true_negatives


def color_target_pred(target, pred, pred_false_pos, xdim_s, ydim_s):
    target_tp = np.zeros(target.shape)
    target_fp = np.zeros(target.shape)
    fill = 3
    if xdim_s != None:
        for xdim, ydim in zip(xdim_s, ydim_s):
            pred_crop = pred[xdim[0]:xdim[1], ydim[0]:ydim[1]]
            pred[xdim[0]:xdim[1], ydim[0]:ydim[1]][pred_crop == 1] = 255
            if np.sum(pred[xdim[0]:xdim[1], ydim[0]:ydim[1]] != 0) > 0:
                target_tp[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 255
                target[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 0
                target_tp[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 255
                target[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 0
                target_tp[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 255
                target[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 0
                target_tp[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 255
                target[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 0
            else:
                target_fp[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 255
                target[xdim[0]:xdim[1], ydim[0]:ydim[0] + fill] = 0
                target_fp[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 255
                target[xdim[0]:xdim[1], ydim[1] - fill:ydim[1]] = 0
                target_fp[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 255
                target[xdim[0]:xdim[0] + fill, ydim[0]:ydim[1]] = 0
                target_fp[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 255
                target[xdim[1] - fill:xdim[1], ydim[0]:ydim[1]] = 0

    pred_false_pos[pred_false_pos == 1] = 255
    pred_rgb = np.dstack((pred_false_pos, pred, np.zeros(pred.shape)))
    target_rgb = np.dstack((target_fp, target_tp, target * 255))
    return target_rgb, pred_rgb