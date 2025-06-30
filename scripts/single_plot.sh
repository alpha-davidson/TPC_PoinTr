#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "single_plot"               # name
#SBATCH --output "plot-out.log"      # output file
#SBATCH --error "plot-err.log"       # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu


source /opt/conda/bin/activate env1

export TORCH_CUDA_ARCH_LIST="8.6"

python graph.py

