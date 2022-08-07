#!/bin/sh
#SBATCH -J ltp
#SBATCH --partition=cpu1
#SBATCH -c 8
#SBATCH -w node22


source /home/jliu288/.bashrc
conda activate cuda113

python label_transfer_plus.py