#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "lossplot"               # name
#SBATCH --output "lossplot-out.log"      # output file
#SBATCH --error "lossplot-err.log"       # error message file

# resource request info 
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu


source /opt/conda/bin/activate env1

export TORCH_CUDA_ARCH_LIST="8.6"

# Might be different, check
EXPERIMENT_PATH="experiments/ALPHA/ALPHA_ATTPC/ATTPC"


# Chooses latest done log
LOG_FILE=$(ls -1 "${EXPERIMENT_PATH}"/[0-9]*.log | sort -t '_' -k1,1 -k2,2 | tail -n 1)

python tools/loss_vs_epoch.py \
  --log_file "${LOG_FILE}" \
  --output "inference_result/loss_vs_epoch.png"
