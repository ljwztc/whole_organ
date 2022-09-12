import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import ceil
from scipy.ndimage.filters import gaussian_filter
import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize, Compose
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    optional_import,
)

from monai.data import decollate_batch
from monai.transforms import Invertd, SaveImaged

TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '01_2': [1,3,4,5,6,7,11,14],
    '02': [1,3,4,5,6,7,11,14],
    '03': [6],
    '04': [6,27], # post process
    '05': [2,3,26], # post process
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,13,20,21,23,24],
    '08': [6, 2, 3, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,4,16,17,2,3],  
    '13': [6,2,3,1,11,8,9,7,4,5,12,13,25], 
    '14': [11, 28],
    '10_03': [6, 27], # post process
    '10_06': [30],
    '10_07': [11, 28], # post process
    '10_08': [15, 29], # post process
    '10_09': [1],
    '10_10': [31]
}

ORGAN_NAME = ['Spleen', 'R Kidney', 'L Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'postcava', 'Portal vein and splenic vein',
                'Pancreas', 'R Adrenal Gland', 'L Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'R Lung', 'L Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'L head of femur', 'R head of femur', 'celiac truck',
                'kidney tumor', 'liver tumor', 'pancreas tumor', 'Hepatic Vessel tumor', 'Lung tumor', 'Colon tumor']

## mapping to original setting
MERGE_MAPPING_v1 = {
    '01': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14)],
    '02': [(1,1), (3,3), (4,4), (5,5), (6,6), (7,7), (11,11), (14,14)],
    '03': [(6,1)],
    '04': [(6,1), (27,2)],
    '05': [(2,1), (3,1), (26, 2)],
    '07': [(1,2), (2,4), (3,3), (4,6), (5,7), (6,1), (7,5), (11,8), (12,12), (13,12), (14,9), (18,10), (19,11), (20,13), (21,14), (23,15), (24,16)],
    '08': [(1,3), (2,2), (6,1), (11,4)],
    '09': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (11,10), (12,11), (13,12), (14,13), (21,14), (22,15)],
    '10_03': [(6,1), (26,2)],
    '10_06': [(30,1)],
    '10_07': [(11,1), (28,2)],
    '10_08': [(15,1), (29,2)],
    '10_09': [(1,1)],
    '10_10': [(31,1)],
    '12': [(2,4), (4,2), (6,1), (16,3), (17,3)],  
    '13': [(1,3), (2,2), (4,8), (5,9), (6,1), (7,7), (8,5), (9,6), (11,4), (12,10), (13,11), (25,12)],

def visualize_label(batch, save_dir, input_transform):
    ### function: save the prediction result into dir
    ## Input
    ## batch: the batch dict output from the monai dataloader
    ## one_channel_label: the predicted reuslt with same shape as label
    ## save_dir: the directory for saving
    ## input_transform: the dataloader transform
    post_transforms = Compose([
        Invertd(
            keys=["label", 'one_channel_label'],
            transform=input_transform,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True,
        ),
        SaveImaged(keys="label", 
                meta_keys="label_meta_dict" , 
                output_dir=save_dir, 
                output_postfix="original", 
                resample=False
        ),
        SaveImaged(keys='one_channel_label', 
                meta_keys="label_meta_dict" , 
                output_dir=save_dir, 
                output_postfix="result", 
                resample=False
        ),
    ])
    
    batch = [post_transforms(i) for i in decollate_batch(batch)]


def merge_label(pred_sigmoid, name):
    B, C, W, H, D = pred_sigmoid.shape
    merged_label = torch.zeros(B,1,W,H,D).cuda()
    pred_bmask = torch.where(pred_sigmoid > 0.5, 1., 0.)
    for b in range(B):
        template_key = get_key(name[b])
        transfer_mapping = MERGE_MAPPING_v1[template_key]
        organ_index = []
        for item in transfer_mapping:
            src, tgt = item
            merged_label[b][0][pred_bmask[b][src-1]==1] = tgt
            # organ_index.append(src-1)
        # organ_index = torch.tensor(organ_index).cuda()
        # predicted_prob = pred_sigmoid[b][organ_index]
    return merged_label



def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def multi_net(net_list, img, task_id):
    # img = torch.from_numpy(img).cuda()

    padded_prediction = net_list[0](img, task_id)
    padded_prediction = F.sigmoid(padded_prediction)
    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img, task_id)
        padded_prediction_i = F.sigmoid(padded_prediction_i)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction#.cpu().data.numpy()

def check_data(dataset_check):
    img = dataset_check[0]["image"]
    label = dataset_check[0]["label"]
    print(dataset_check[0]["name"])
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print(torch.unique(label[0, :, :, 150]))
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, 150].detach().cpu(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, 150].detach().cpu())
    plt.show()
