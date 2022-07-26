import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model.SwinUNETR_partial import SwinUNETR
from dataset.dataloader import get_loader
from dataset.utils import TEMPLATE
from utils import loss
from utils.utils import predict_sliding, dice_score, check_data, generate_label


torch.multiprocessing.set_sharing_strategy('file_system')

NUM_CLASS = 31

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    eval_num = 30
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        print(batch.keys())
        x, y, name = batch["image"].to(args.device), batch["label"].to(args.device), batch['name']
        if args.dist:
            dist.barrier()
        logit_map = model(x)
        # print(name, model.organ_embedding.weight[2])
        # print(torch.unique(y))
        y = generate_label(y, NUM_CLASS, name, TEMPLATE)
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y)
        loss = term_seg_Dice + term_seg_BCE
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        if (args.epoch % args.store_num == 0 and args.epoch != 0) and dist.get_rank() == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir("out/"):
                os.mkdir("out/")
            torch.save(checkpoint, 'out/' + 'epoch_' + str(args.epoch) + '.pth')
            print('save model success')
        torch.cuda.empty_cache()

def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

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
    model_dict = torch.load(args.pretrain)["state_dict"]
    for key in model_dict.keys():
        if 'out' not in key:
            store_dict[key] = model_dict[key]

    model.load_state_dict(store_dict)
    print('Use pretrained weights')

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint)
        else:
            store_dict = model.state_dict()
            for key in checkpoint.keys():
                store_dict['.'.join(key.split('.')[1:])] = checkpoint[key]
            model.load_state_dict(store_dict)
        # model.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # args.epoch = checkpoint['epoch']
        print('success resume from ', args.resume)

    model.to(args.device)
    model.train()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])

    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    torch.backends.cudnn.benchmark = True


    train_loader, train_sampler, _ = get_loader(args)

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
        args.epoch += 1
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, help='Number of training epoches')
    parser.add_argument('--store_num', default=30, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['organ_plus']) # 'organ_plusplus', 'organ_plus', 'single_organ', 'mri'
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='/home/jliu288/code/OCDA/multi-organ/unified_organ/dataset/whole_oragn/', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=8, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=0.8, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=0.8, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=64, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    args = parser.parse_args()

    process(args=args)

if __name__ == "__main__":
    main()

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train_cond.py --dist True