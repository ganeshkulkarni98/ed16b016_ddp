#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -q gpuq
#PBS -e errorfile_5.err
#PBS -o logfile_5.log
#PBS -l select=1:ncpus=1:ngpus=1
conda init bash
module load anaconda3_2020
source activate kanhiya
~/.conda/envs/kanhiya/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/yolov5/train.py'
source deactivate
