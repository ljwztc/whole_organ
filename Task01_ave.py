import glob

txt_dir = './out/Nvidia/old_fold0'
for i in range(450, 511, 10):
    organ_dice = 0
    for line in open(txt_dir+ f'/val_{i}.txt'):
        if line[:7] == 'Task01|':
            organ_list = line.split(',')
            for j in range(13):
                organ_dice += float(organ_list[j][-6:])
    print(i, organ_dice/13)