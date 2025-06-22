#!/bin/bash
#
### Job Parameters:
# basic info
#SBATCH --job-name "create_env"          # name
#SBATCH --output "create_env-out.log"    # output file
#SBATCH --error "create_env-err.log"     # error message file

# resource request info 
#SBATCH --constraint cuda11 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu

conda create -n env1 python=3.9 -c pytorch -c defaults
source /opt/conda/bin/activate env1

conda install -n env1 -c pytorch \
               pytorch=1.13.1 \
               torchvision=0.14.1 \
               torchaudio=0.13.1 \
               cudatoolkit=11.6 \
               mkl=2021.4 \
               intel-openmp=2021.4
               
# New:
conda install -c nvidia cuda-compiler=11.7
conda install -c conda-forge gxx_linux-64 ninja    



pip3 install -r requirements.txt