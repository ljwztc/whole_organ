#!/bin/sh
#SBATCH -J ltp
#SBATCH --partition=cpu1
#SBATCH -c 16
#SBATCH -w node22

source /home/jliu288/.bashrc
conda activate cuda113

python label_transfer.py