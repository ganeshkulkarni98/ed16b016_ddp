#!/bin/bash
#PBS -l walltime=23:59:00
#PBS -q gpuq
#PBS -e errorfile_4.err
#PBS -o logfile_4.log
#PBS -l select=1:ncpus=2:ngpus=1
conda init bash
module load anaconda3_2020
conda activate myenv
~/.conda/envs/myenv/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/fasterrcnn/FasterRCNN/train_4.py'
conda deactivate
