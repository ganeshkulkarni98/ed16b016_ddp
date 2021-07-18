#!/bin/bash
#PBS -l walltime=23:59:00
#PBS -q gpuq
#PBS -e errorfile_2.err
#PBS -o logfile_2.log
#PBS -l select=1:ncpus=4:ngpus=1
conda init bash
module load anaconda3_2020
conda activate myenv
~/.conda/envs/myenv/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/fasterrcnn/FasterRCNN/train_1.py'
conda deactivate
