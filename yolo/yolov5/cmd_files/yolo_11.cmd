#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -q gpuq
#PBS -e errorfile_7.err
#PBS -o logfile_7.log
#PBS -l select=1:ncpus=1:ngpus=1
conda init bash
module load anaconda3_2020
conda activate ganu
~/.conda/envs/ganu/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/test/yolov5/train.py'
conda deactivate
