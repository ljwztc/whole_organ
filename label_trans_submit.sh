#!/bin/sh
#SBATCH -J ltp
#SBATCH --partition=cpu1
#SBATCH -c 32
#SBATCH -w node23

source /home/jliu288/.bashrc
conda activate cuda113

python label_transfer.py