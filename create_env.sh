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


conda create -n env1 python=3.8 cudatoolkit=11.1 -c conda-forge
source /opt/conda/bin/activate env1


pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install -r requirements.txt

