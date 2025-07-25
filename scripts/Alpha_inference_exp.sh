#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "inference and plot experimental"               # name
#SBATCH --output "inference and plot experimental-out.log"      # output file
#SBATCH --error "inference and plot experimental-err.log"       # error message file

# resource request info 
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu


source /opt/conda/bin/activate env1

export TORCH_CUDA_ARCH_LIST="8.6"


# To choose random partial point cloud and its gt:
DATA_ROOT="/home/DAVIDSON/hayavuzkara/Data/22MgExp/all"


RAND_PATH="$(find "${DATA_ROOT}" | shuf -n 1)"
RAND_EVENT_NAME="$(basename "${RAND_PATH}" .npy)"


# Inference
# Path switched
python tools/inference.py cfgs/ALPHA_ATTPC/ALPHA.yaml \
  experiments/ALPHA/ALPHA_ATTPC/ATTPC/MG22Only/ckpt-best.pth \
  --pc "${RAND_PATH}" --out_pc_root "inference_result/Example_pc" --pc_name "${RAND_EVENT_NAME}"


# Graph 
python tools/graph.py \
  --partial_path "${RAND_PATH}" \
  --predict_path "inference_result/Example_pc/${RAND_EVENT_NAME}_fine.npy"