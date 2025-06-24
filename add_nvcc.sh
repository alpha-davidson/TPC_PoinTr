#!/bin/bash
#
### Job Parameters:
# basic info
#SBATCH --job-name "nvcc"               # name
#SBATCH --output "nvcc_out"      # output file
#SBATCH --error "nvcc_err"       # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu
#SBATCH --constraint cuda11

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin