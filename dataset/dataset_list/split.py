from random import sample

file_name = 'PAOT'
with open(file_name + '.txt', 'r') as f:
    line_set=[]
    lines = f.readlines()
    for line in lines:
        line_set.append(line)
line_set = sorted(line_set)

train_set = sample(line_set, int(len(line_set)*0.7))
train_set = sorted(train_set)
with open(file_name + '_train.txt', 'w') as f:
    for line in train_set:
        f.write(line)

rest_set = []
for line in line_set:
    if line in train_set:
        pass
    else:
        rest_set.append(line)

val_set = sample(rest_set, int(len(rest_set)*0.35))
val_set = sorted(val_set)

with open(file_name + '_val.txt', 'w') as f:
    for line in val_set:
        f.write(line)

with open(file_name + '_test.txt', 'w') as f:
    for line in rest_set:
        if line in val_set:
            pass
        else:
            f.write(line)