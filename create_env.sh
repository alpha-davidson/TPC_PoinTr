#!/bin/bash
#
### Job Parameters:
# basic info
#SBATCH --job-name "create_env"               # name
#SBATCH --output "create_env-out.log"      # output file
#SBATCH --error "create_env-err.log"       # error message file

# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu
#SBATCH --constraint cuda11
#SBATCH --exclude alpha[0-2]

conda create -n env1 python=3.7
source /opt/conda/bin/activate env1



pip3 install --index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio

pip3 install -r requirements.txt

