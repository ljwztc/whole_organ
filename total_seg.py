import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import glob

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch, list_data_collate, Dataset, DataLoader
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.SwinUNETR_partial import SwinUNETR
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ

torch.multiprocessing.set_sharing_strategy('file_system')

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
from monai.data import decollate_batch
from monai.transforms import Invertd, SaveImaged

test_organ_dic = ['aorta', 'lung', 'adrenal gland', 'inferior vena cava', 'spleen', 'liver', 'gall bladder', 'pancreas', 'kidney',
                 'esophagus', 'stomach', 'duodenum', 'colon']

def validation(model, ValLoader, val_transforms, args):
    save_dir = 'out/' + args.log_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model.eval()
    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, name = batch["image"].cuda(), batch["name"]
        # print(label.shape)
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.25, mode='gaussian', device=torch.device('cpu'))
            pred_sigmoid = F.sigmoid(pred).cuda()
        
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        B = pred_hard.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list)
            pred_hard_post = torch.tensor(pred_hard_post)

        ### testing phase for this function
        one_channel_label_v1, _ = merge_label(pred_hard_post, name)
        batch['one_channel_label_v1'] = one_channel_label_v1.cpu()

        post_transforms = Compose([
            Invertd(
                keys=['one_channel_label_v1'],
                transform=val_transforms,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
            SaveImaged(keys='one_channel_label_v1', 
                    output_dir=save_dir + '/output/' + name[0].split('/')[-1].split('.')[0], 
                    output_postfix="result", 
                    resample=False
            ),
        ])
    
        batch = [post_transforms(i) for i in decollate_batch(batch)]

            
        torch.cuda.empty_cache()




def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='total_seg', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default='./out/PAOT_v2/epoch_430.pth', help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')

    args = parser.parse_args()

    # prepare the 3D model
    model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                      in_channels=1,
                      out_channels=NUM_CLASS,
                      feature_size=48,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=False,
                     )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    args.epoch = checkpoint['epoch']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    model.load_state_dict(store_dict)
    print(f'Use {args.resume} weights')

    model.cuda()

    torch.backends.cudnn.benchmark = True

    ## for dataset
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )

    ## test dict part
    data_txt = 'total_seg/val.txt'
    
    test_dir = '/home/jliu288/data/whole_organ/Totalsegmentator_dataset/'
    test_img = []
    test_name = []
    for line in open(data_txt):
        test_img.append(test_dir + line.split(' ')[0] + '/ct.nii.gz')
        test_name.append('15' + line.split(' ')[0])
    data_dicts_test = [{'image': image, 'name': name}
                for image, name in zip(test_img, test_name)]
    print('test len {}'.format(len(data_dicts_test)))


    test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)


    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
