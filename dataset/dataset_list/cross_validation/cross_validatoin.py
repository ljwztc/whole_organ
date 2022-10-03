import random
import numpy as np

data_set = []
for line in open('PAOT.txt'):
    data_set.append(line)

random_list = random.sample(range(len(data_set)), len(data_set))

dataset_sub = []
interval = int(len(data_set)/5)
for i in range(5):
    dataset_sub.append([])
    for j in range(i*interval,(i+1)*interval):
        dataset_sub[i].append(data_set[random_list[j]])

for i in range(5):
    val_dataset = dataset_sub[i]
    val_dataset = sorted(val_dataset)
    with open(f'PAOT_val_{i}.txt', 'w') as f:
        for item in val_dataset:
            f.write(item)
    with open(f'PAOT_test_{i}.txt', 'w') as f:
        for item in val_dataset:
            f.write(item)
    with open(f'PAOT_train_{i}.txt', 'w') as f:
        for item in data_set:
            if item not in val_dataset:
                f.write(item)