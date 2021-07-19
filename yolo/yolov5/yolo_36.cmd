#!/bin/bash
#PBS -l walltime=23:59:00
#PBS -q gpuq
#PBS -e errorfile_36.err
#PBS -o logfile_36.log
#PBS -l select=1:ncpus=2:ngpus=1
conda init bash
module load anaconda3_2020
conda activate ganu
~/.conda/envs/ganu/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/yolo/yolov5/train_36.py'
conda deactivate
