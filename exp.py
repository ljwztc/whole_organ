import numpy as np
from utils.utils import dice_score
import torch
from utils.utils import extract_topk_largest_candidates, organ_post_process

organ_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'R Lung', 'L Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 'Colon Tumor', 'Kidney Cyst']

data = np.load('out/PAOT/test_320/predict/01_Multi-Atlas_Labelinglabel0001.npz')
pred, label = data['pred'], data['label']

pred = pred / 255

pred_hard = pred > 0.5
pred_hard = pred_hard[0]

pred_hard_post = organ_post_process(pred_hard, organ_list)


for organ in organ_list:
    if np.sum(label[0,organ-1,:,:,:]) != 0:
        dice_organ, recall, precision = dice_score(torch.tensor(pred_hard[organ-1,:,:,:]), torch.tensor(label[0,organ-1,:,:,:]))
        print(dice_organ, recall, precision)
        dice_organ, recall, precision = dice_score(torch.tensor(pred_hard_post[organ-1,:,:,:]), torch.tensor(label[0,organ-1,:,:,:]))
        print(dice_organ, recall, precision, ORGAN_NAME[organ-1])
