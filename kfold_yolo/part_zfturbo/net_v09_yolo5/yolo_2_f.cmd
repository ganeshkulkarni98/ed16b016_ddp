#!/bin/bash
#PBS -l walltime=23:59:00
#PBS -q gpuq
#PBS -e errorfile_2_f.err
#PBS -o logfile_2_f.log
#PBS -l select=1:ncpus=2:ngpus=1
conda init bash
module load anaconda3_2020
conda activate ganu
~/.conda/envs/ganu/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/kfold_yolo/part_zfturbo/net_v09_yolo5/train_2_f.py'
conda deactivate
