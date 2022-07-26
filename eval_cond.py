import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model.SwinUNETR_partial import SwinUNETR
from dataset.dataloader import mri_dataloader, ct_dataloader, partial_label_dataloader, partial_label_test_dataloader
from utils import loss
from utils.utils import predict_sliding, dice_score, sliding_window_inference

torch.multiprocessing.set_sharing_strategy('file_system')

NUM_CLASS = 2

# prepare the 3D model
model = SwinUNETR(img_size=(64, 64, 64),
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
load_dict = torch.load('./out/epoch_330.pth')

for key, value in load_dict.items():
    name = '.'.join(key.split('.')[1:])
    store_dict[name] = value

model.load_state_dict(store_dict)
print('Use pretrained weights')

model.cuda()
model.eval()

# criterion and optimizer
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True

test_loader = partial_label_test_dataloader()


def validation(ValLoader):
    model.eval()
    val_Dice = torch.zeros(size=(7, 2)).cuda()  # np.zeros(shape=(7, 2))
    count = torch.zeros(size=(7, 2)).cuda()  # np.zeros(shape=(7, 2))

    for index, batch in enumerate(ValLoader):
        # print('%d processd' % (index))
        image, label, name, task_id = batch["image"].cuda(), batch["label"].cuda(), batch["name"], batch["task_id"].cuda()
        
        with torch.no_grad():
            pred = sliding_window_inference(image, (192, 192, 64), 1, model, task_id)
            pred_sigmoid = F.sigmoid(pred)
            # print(pred_sigmoid.shape, label.shape)
            # pred_sigmoid = predict_sliding([model], image.numpy(), [128, 128, 128], NUM_CLASS, task_id)

            if label[0, 0, 0, 0, 0] == -1:
                dice_c1 = torch.from_numpy(np.array([-999]))
            else:
                dice_c1 = dice_score(pred_sigmoid[:, 0, :, :, :], label[:, 0, :, :, :])
                val_Dice[task_id[0], 0] += dice_c1
                count[task_id[0], 0] += 1
            if label[0, 1, 0, 0, 0] == -1:
                dice_c2 = torch.from_numpy(np.array([-999]))
            else:
                dice_c2 = dice_score(pred_sigmoid[:, 1, :, :, :], label[:, 1, :, :, :])
                val_Dice[task_id[0], 1] += dice_c2
                count[task_id[0], 1] += 1
        
        print('Task%d-%s Organ:%.4f Tumor:%.4f' % (task_id, name, dice_c1.item(), dice_c2.item()))


    count[count == 0] = 1
    val_Dice = val_Dice / count

    for t in range(7):
        print('Sum: Task%d- Organ:%.4f Tumor:%.4f' % (t, val_Dice[t, 0], val_Dice[t, 1]))
    
    mdice = (np.mean(val_Dice[:6, 1])*6+np.mean(val_Dice[:4, 0])*4+val_Dice[6, 0])/11

    model.train()
    return mdice


max_iterations = 100000
eval_num = 5000
post_label = AsDiscrete(to_onehot=NUM_CLASS)
post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASS)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best1 = 0
global_step_best1 = 0
metric_values1 = []
metric_values2 = []

validation(test_loader)

# torch.save(model.state_dict(), os.path.join('./out/', "dataset_mri_trans.pth"))
# metric_values1 = np.array(metric_values1)
# np.save('./out/mri_ct_trans300.npy', metric_values1)

# metric_values2 = np.array(metric_values2)
# np.save('./out/mri_mri_trans300.npy', metric_values2)