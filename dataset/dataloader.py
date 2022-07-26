from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy


import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler

from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor

from .utils import TEMPLATE, rl_split


class ToTemplatelabel(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, totemplate: List) -> NdarrayOrTensor:
        new_lbl = np.zeros(lbl.shape)
        for src, tgt in enumerate(totemplate):
            new_lbl[lbl == (src+1)] = tgt
        # unique,count=np.unique(new_lbl,return_counts=True)
        # data_count=dict(zip(unique,count))
        # print(data_count)
        # unique,count=np.unique(lbl,return_counts=True)
        # data_count=dict(zip(unique,count))
        # print(data_count)
        return new_lbl

class ToTemplatelabeld(MapTransform):
    backend = ToTemplatelabel.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.totemplate = ToTemplatelabel()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_index = int(d['name'][0:2])
        if dataset_index == 1 or dataset_index == 2:
            pass
        else:
            template_key = d['name'][0:2]
            d['label'] = self.totemplate(d['label'], TEMPLATE[template_key])
        return d

class RL_Split(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, organ_list: List, name) -> NdarrayOrTensor:
        lbl_new = lbl.copy()
        for organ in organ_list:
            organ_index = organ
            right_index = organ
            left_index = organ + 1
            lbl_post = rl_split(lbl_new[0], organ_index, right_index, left_index, name)
            lbl_new[lbl_post == left_index] = left_index
        return lbl_new

class RL_Splitd(MapTransform):
    backend = ToTemplatelabel.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spliter = RL_Split()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_index = int(d['name'][0:2])
        # print(d['name'], dataset_index)
        if dataset_index in [5,8,13]:
            # print(d['name'], np.unique(d['label']))
            d['label'] = self.spliter(d['label'], [2], d['name'])
            # print(d['name'], np.unique(d['label']))
        elif dataset_index == 7:
            d['label'] = self.spliter(d['label'], [12], d['name'])
        elif dataset_index == 12:
            d['label'] = self.spliter(d['label'], [2, 16], d['name'])
        else:
            pass
        return d

def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ToTemplatelabeld(keys=['label']),
            RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ToTemplatelabeld(keys=['label']),
            RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_img = []
    train_lbl = []
    train_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_train.txt'):
            train_img.append(args.data_root_path + line.strip().split()[0])
            train_lbl.append(args.data_root_path + line.strip().split()[1])
            train_name.append(line.strip().split()[1].split('.')[0])
    data_dicts_train = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(train_img, train_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))

    train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                collate_fn=list_data_collate, sampler=train_sampler)
    
    test_img = []
    test_lbl = []
    test_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_test.txt'):
            test_img.append(args.data_root_path + line.strip().split()[0])
            test_lbl.append(args.data_root_path + line.strip().split()[1])
            test_name.append(line.strip().split()[1].split('.')[0])
    data_dicts_test = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(test_img, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)


    return train_loader, train_sampler, test_loader

if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()