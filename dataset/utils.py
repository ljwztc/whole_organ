#import cc3d # pip install connected-components-3d
import numpy as np

TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '02': [1,3,4,5,6,7,11,14],
    '03': [6],
    '04': [6,27],
    '05': [2,26],
    '07': [6,1,3,2,7,4,5,11,14,17,18,12,19,20,22,23],
    '08': [6, 2, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,4,16,2],
    '13': [6,2,1,11,8,9,7,4,5,12,13,25],
    '10_03': [6, 27],
    '10_06': [30],
    '10_07': [11, 28],
    '10_08': [15, 29],
    '10_09': [1],
    '10_10': [31]
}

def rl_split(input_data, organ_index, right_index, left_index, name):
    '''
    input_data: 3-d tensor [w,h,d], after transform 'Orientationd(keys=["label"], axcodes="RAS")'
    oragn_index: the organ index of interest
    right_index and left_index: the corresponding index in template
    return [1, w, h, d]
    '''
    RIGHT_ORGAN = right_index
    LEFT_ORGAN = left_index
    label_raw = input_data.copy()
    label_in = np.zeros(label_raw.shape)
    label_in[label_raw == organ_index] = 1
    
    label_out = cc3d.connected_components(label_in, connectivity=26)
    # print('label_out', organ_index, np.unique(label_out), np.unique(label_in), label_out.shape, np.sum(label_raw == organ_index))
    # assert len(np.unique(label_out)) == 3, f'more than 2 component in this ct for {name} with {np.unique(label_out)} component'
    if len(np.unique(label_out)) > 3:
        count_sum = 0
        values, counts = np.unique(label_out, return_counts=True)
        num_list_sorted = sorted(values, key=lambda x: counts[x])[::-1]
        for i in num_list_sorted[3:]:
            label_out[label_out==i] = 0
            count_sum += counts[i]
        label_new = np.zeros(label_out.shape)
        for tgt, src in enumerate(num_list_sorted[:3]):
            label_new[label_out==src] = tgt
        label_out = label_new
        print(f'In {name}. Delete {len(num_list_sorted[3:])} small regions with {count_sum} voxels')
    a1,b1,c1 = np.where(label_out==1)
    a2,b2,c2 = np.where(label_out==2)
    
    label_new = np.zeros(label_out.shape)
    if np.mean(a1) < np.mean(a2):
        label_new[label_out==1] = LEFT_ORGAN
        label_new[label_out==2] = RIGHT_ORGAN
    else:
        label_new[label_out==1] = RIGHT_ORGAN
        label_new[label_out==2] = LEFT_ORGAN
    
    return label_new[None]