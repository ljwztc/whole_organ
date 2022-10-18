dataset_dic = {'train': [], 'test': [], 'val': []}
sub_set = ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']
for set_index in ['train', 'test', 'val']:
    for sub_index in sub_set:
        with open(f'PAOT_{sub_index}_{set_index}.txt', 'w') as f:
            for line in open(f'PAOT_tumor_{set_index}.txt'):
                if (line[0:2] + '_' + line[17:19]) == sub_index:
                    f.write(line)
