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
from utils.utils import dice_score, TEMPLATE, ORGAN_NAME

torch.multiprocessing.set_sharing_strategy('file_system')

NUM_CLASS = 31


def validation(model, ValLoader, args):
    save_dir = 'out/' + args.log_name + f'/test_{args.epoch}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir+'/predict')
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        # print(label.shape)
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
            # print(pred_sigmoid.shape, label.shape)
        if args.store_result:
            pred_sigmoid_store = (pred_sigmoid.cpu().numpy() * 255).astype(np.uint8)
            label_store = (label.numpy()).astype(np.uint8)
            np.savez_compressed(save_dir + '/predict/' + name[0].split('/')[0] + name[0].split('/')[-1], 
                            pred=pred_sigmoid_store, label=label_store)
            ## load data
            # data = np.load('/out/epoch_80/predict/****.npz')
            # pred, label = data['pred'], data['label']
        
        B = pred_sigmoid.shape[0]
        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]
            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                if torch.sum(label[b,organ-1,:,:,:].cuda()) != 0:
                    dice_organ = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open('out/'+args.log_name+f'/test_{args.epoch}.txt', 'w') as f:
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
        
    
    np.save(save_dir + '/result.npy', dice_list)
    # load
    # dice_list = np.load(/out/epoch_xxx/result.npy, allow_pickle=True)




def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='PAOT', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default='./out/PAOT/epoch_100.pth', help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT']) # 'PAOT', 'felix'
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
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')

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
    print('Use pretrained weights')

    model.cuda()

    torch.backends.cudnn.benchmark = True

    _, _, _, test_loader = get_loader(args)

    validation(model, test_loader, args)

if __name__ == "__main__":
    main()
