#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -q gpuq
#PBS -e errorfile_6.err
#PBS -o logfile_6.log
#PBS -l select=1:ncpus=1:ngpus=1
conda init bash
module load anaconda3_2020
source activate ganu
~/.conda/envs/ganu/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/yolov5/train.py'
source deactivate
