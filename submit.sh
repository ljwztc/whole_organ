#!/bin/sh
#SBATCH -J part_t
#SBATCH --partition=team1
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH -w node31
#SBATCH -o out/partial_label.out

EXP=singlecard_adv
PORT=1205
GPUS=2

source /home/jliu288/.bashrc
nvidia-smi
conda activate cuda113

#python 
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train_adv.py --snapshot-dir ./snapshots/$EXP --log-dir ./log/$EXP  --port $PORT
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1105 train_so.py --snapshot-dir ./snapshots/try --log-dir ./log/try  --port 1105
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT train.py --dist True