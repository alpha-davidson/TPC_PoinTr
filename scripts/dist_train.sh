#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "train"               # name
#SBATCH --output "train-out.log"      # output file
#SBATCH --error "train-err.log"       # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu

source /opt/conda/bin/activate env1

python main.py --config cfgs/PCN_models/AdaPoinTr.yaml --exp_name example