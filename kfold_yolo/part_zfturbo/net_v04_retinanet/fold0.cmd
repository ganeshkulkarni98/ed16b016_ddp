#!/bin/bash
#PBS -l walltime=23:59:00
#PBS -q gpuq
#PBS -e errorfile_0.err
#PBS -o logfile_0.log
#PBS -l select=1:ncpus=2:ngpus=1
conda init bash
module load anaconda3_2020
conda activate tf-gpu
~/.conda/envs/tf-gpu/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/project/kfold_yolo/part_zfturbo/net_v04_retinanet/r21_train_backbone_resnet101_sqr_fold0.py'
conda deactivate
