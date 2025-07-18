#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "CHECK_GPU"               # name
#SBATCH --output "CHECK_GPU-out.log"      # output file
#SBATCH --error "CHECK_GPU-err.log"       # error message file

# resource request info 
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu

nvidia-smi

source /opt/conda/bin/activate env1

export TORCH_CUDA_ARCH_LIST="8.6"

python check.py
