#!/bin/bash
#PBS -l walltime=05:00:00
#PBS -q gpuq
#PBS -e errorfile_3.err
#PBS -o logfile_3.log
#PBS -l select=1:ncpus=16:ngpus=1
conda init bash
module load anaconda3_2020
source activate myenv
~/.conda/envs/myenv/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/yolov5/train.py'
source deactivate
