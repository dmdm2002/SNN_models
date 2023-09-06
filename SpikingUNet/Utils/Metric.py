"""https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-with-pytorch/blob/master/IOU.py"""

import torch
import torch.nn.functional as F
import numpy as np


def get_iou(pred, gt, classes_name, display=False):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).view(-1)
    gt = gt.view(-1)

    class_iou = np.zeros(classes_name)
    class_weight = np.zeros(classes_name)
    pixel_acc = np.float32(np.sum(pred == gt)) / gt.size

    for i in range(classes_name):
        intersection = np.float32(np.sum((pred == gt) * (gt == i)))
        union = np.sum(gt == i) + np.sum(pred == i) - intersection
        if union > 0:
            class_iou[i] = intersection / union
            class_weight[i] = union

    if display:
        for i in range(len(classes_name)):
            print(f'{classes_name[i]} IoU: {class_iou[i]}')
        print(f'Mean Classes IOU: {np.mean(class_iou)}')
        print(f'Pixel Accuracy: {pixel_acc}')

    return class_iou, class_weight
