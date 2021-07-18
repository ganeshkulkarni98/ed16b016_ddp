#!/bin/bash
#PBS -l walltime=23:59:00
#PBS -q gpuq
#PBS -e errorfile_1.err
#PBS -o logfile_1.log
#PBS -l select=1:ncpus=2:ngpus=1
conda init bash
module load anaconda3_2020
conda activate ganesh
~/.conda/envs/ganesh/bin/python3 '/lfs/usrhome/btech/ed16b016/scratch/fasterrcnn/detectron2/det_train.py'
conda deactivate
