#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "check_install"               # name
#SBATCH --output "check_install.log"      # output file
#SBATCH --error "check_install.log"       # error message file


# resource request info 
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu
#SBATCH --constraint cuda11

source /opt/conda/bin/activate env1
python tools/installation_check.py