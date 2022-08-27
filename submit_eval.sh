#!/bin/sh
#SBATCH -J 025_gau
#SBATCH --partition=team1
#SBATCH --gres=gpu:1
#SBATCH -c 3
#SBATCH -w node31
#SBATCH -o out/025_gau.out

EXP=singlecard_adv
PORT=1105
GPUS=2

source /home/jliu288/.bashrc
nvidia-smi
conda activate cuda113

#python 
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train_adv.py --snapshot-dir ./snapshots/$EXP --log-dir ./log/$EXP  --port $PORT
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1105 train_so.py --snapshot-dir ./snapshots/try --log-dir ./log/try  --port 1105
python debug_025_gaussian.py