#!/bin/bash
#PBS -l walltime=23:59:00
#PBS -q gpuq
#PBS -e errorfile_24_f2.err
#PBS -o logfile_24_f2.log
#PBS -l select=1:ncpus=2:ngpus=1
conda init bash
module load anaconda3_2020
conda activate ganu
~/.conda/envs/ganu/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/yolov5/train_24_further.py'
conda deactivate
