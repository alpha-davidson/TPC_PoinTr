#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "train_A"               # name
#SBATCH --output "train_A-out.log"      # output file
#SBATCH --error "train_A-err.log"       # error message file

# resource request info 
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu




source /opt/conda/bin/activate env1

export TORCH_CUDA_ARCH_LIST="8.6"


python main.py --config cfgs/ALPHA_ATTPC/ALPHA.yaml --exp_name ATTPC