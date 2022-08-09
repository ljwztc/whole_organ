## Data process
1. Download data to ORGAN_DATASET_DIR.  
2. Modify the ORGAN_DATASET_DIR value in label_transfer.py (line 47) and NUM_WORKER(line 50)
3. ```python label_transfer.py```

## Installation

```
pip install -r requirements.txt
pip install 'monai[all]'
```

## Train
```python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --dist True --data_root_path /mnt/medical_data/PublicAbdominalData/```

## Test
```python eval.py --resume ./out/epoch_330.pth```
