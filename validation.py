import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.SwinUNETR_partial import SwinUNETR
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, TEMPLATE, ORGAN_NAME, merge_label, visualize_label, get_key, NUM_CLASS
from utils.utils import extract_topk_largest_candidates, organ_post_process, threshold_organ

torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, args, i):
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        # print(name, label.shape)
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()
        B = pred_sigmoid.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list)
            pred_hard_post = torch.tensor(pred_hard_post)

            for organ in organ_list:
                if torch.sum(label[b,organ-1,:,:,:]) != 0:
                    dice_organ, _, _ = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open('out/'+args.log_name+f'/val_{i}.txt', 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            for organ in organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
            print(content)
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in range(NUM_CLASS):
            content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
        print(content)
        f.write(content)
        f.write('\n')
        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write('%s: %.4f, '%('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
        f.write('\n')




def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='PAOT_v2', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--start_epoch', default=410, type=int, help='Number of start epoches')
    parser.add_argument('--end_epoch', default=450, type=int, help='Number of end epoches')
    parser.add_argument('--epoch_interval', default=10, type=int, help='Number of start epoches')

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

    parser.add_argument('--phase', default='validation', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')

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
    
    for i in range(args.start_epoch, args.end_epoch+1, args.epoch_interval):
        #Load pre-trained weights
        store_path = f'out/{args.log_name}/epoch_{i}.pth'
        store_dict = model.state_dict()
        load_dict = torch.load(store_path)['net']

        for key, value in load_dict.items():
            name = '.'.join(key.split('.')[1:])
            store_dict[name] = value

        model.load_state_dict(store_dict)
        print(f'Load epoch_{i} weights')

        model.cuda()

        torch.backends.cudnn.benchmark = True

        validation_loader, val_transforms = get_loader(args)

        validation(model, validation_loader, args, i)

if __name__ == "__main__":
    main()
