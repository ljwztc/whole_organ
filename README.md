## Installation

```
git clone https://github.com/ljwztc/whole_organ.git
cd whole_organ/
pip install -r requirements.txt
pip install 'monai[all]'
cd pretrained_weights/
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
cd ../
```

## Data process
1. Download data to ORGAN_DATASET_DIR (referring to [Genesis Abdomen](https://github.com/MrGiovanni/GenesisLung#1-download-data-assembly-1)).
2. Modify the ORGAN_DATASET_DIR value in label_transfer.py (line 47) and NUM_WORKER (line 50)
3. ```python -W ignore label_transfer.py```


Check README in pretrained_weights

## Train
```
CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node=6 --master_port=1234 train.py --dist True --data_root_path /mnt/medical_data/PublicAbdominalData/ --resume out/epoch_10.pth --num_workers 12 --num_samples 4
```

## Test
```
CUDA_VISIBLE_DEVICES=7 python -W ignore eval.py --resume ./out/epoch_61.pth --data_root_path /mnt/medical_data/PublicAbdominalData/
```

## FQA

1. How to add new datasets to the respository?
2. 
