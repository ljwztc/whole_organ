## Data process
1. Download data to ORGAN_DATASET_DIR.  
2. Modify the ORGAN_DATASET_DIR value in label_transfer.py (line 47) and NUM_WORKER(line 50)
3. ```python label_transfer.py```

## Installation

```
pip install -r requirements.txt
pip install 'monai[all]'
```
Check README in pretrained_weights

## Train
```
CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port=1234 train.py --dist True --data_root_path /mnt/medical_data/PublicAbdominalData/ --resume out/epoch_10.pth --num_workers 12 --num_samples 4
```

## Test
```
python eval.py --resume ./out/epoch_330.pth
```
