import random
import numpy as np

data_set = {
    '01': [],
    '02': [],
    '03': [],
    '04': [],
    '05': [],
    '07': [],
    '08': [],
    '09': [],
    '12': [],
    '13': []
}
data_key = data_set.keys()

for line in open('PAOT.txt'):
    if line[0:2] in data_key:
        data_set[line[0:2]].append(line)

random_list = {}
for key in data_key:
    random_list[key] = random.sample(range(len(data_set[key])), len(data_set[key]))

dataset_sub = {}
for key in data_key:
    interval = int(len(data_set[key])/5)
    dataset_sub[key] = []
    for i in range(5):
        dataset_sub[key].append([])
        for j in range(i*interval,(i+1)*interval):
            dataset_sub[key][i].append(data_set[key][random_list[key][j]])

for i in range(5):
    for key in data_key:
        val_dataset = dataset_sub[key][i]
        val_dataset = sorted(val_dataset)
        with open(f'PAOT_{i}_val.txt', 'a') as f:
            for item in val_dataset:
                f.write(item)
        with open(f'PAOT_{i}_test.txt', 'a') as f:
            for item in val_dataset:
                f.write(item)
        with open(f'PAOT_{i}_train.txt', 'a') as f:
            data_list = sorted(data_set[key])
            for item in data_list:
                if item not in val_dataset:
                    f.write(item)