import random
import numpy as np
import glob

organ = []
root_dir = '/home/jliu288/data/whole_organ/10_Decathlon/'
for task in glob.glob(root_dir+'Task**'):
    for ct in glob.glob(task+'/imagesTr/**.nii.gz'):
        organ.append(ct.split('/')[-1])


for i in range(5):
    for item in ['train', 'test', 'val']:
        txt_pth = f'PAOT_{i}_{item}.txt'
        for line in open(txt_pth):
            if int(line[:2]) == 10:
                if line.split('\t')[0].split('/')[-1] not in organ:
                    print(i, item, line)
