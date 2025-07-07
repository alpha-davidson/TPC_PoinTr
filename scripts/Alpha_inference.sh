#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "inference and plot"               # name
#SBATCH --output "inference and plot-out.log"      # output file
#SBATCH --error "inference and plot-err.log"       # error message file

# resource request info 
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu


source /opt/conda/bin/activate env1

export TORCH_CUDA_ARCH_LIST="8.6"

# Inference
python tools/inference.py cfgs/ALPHA_ATTPC/ALPHA.yaml \
  experiments/ALPHA/ALPHA_ATTPC/ATTPC/ckpt-best.pth \
  --pc_root """demo/ALPHA""" --save_vis_img --out_pc_root inference_result/

# demo/ALPHA must be adjusted to get a random or adjusted one from the dataset

# Maybe run the graph from here also, considering the locations of the partial, created, and ground truth. Still done for simulated data and will be later tried on experimental data.


# Loss vs Epoch plot
EXPERIMENT_PATH="experiments/ALPHA/ALPHA_ATTPC/ATTPC"


# Chooses latest done log
LOG_FILE=$(ls -1 "${EXPERIMENT_PATH}"/[0-9]*.log | sort -t '_' -k1,1 -k2,2 | tail -n 1)

python tools/loss_vs_epoch.py \
  --log_file "${LOG_FILE}" \
  --output "inference_result/loss_vs_epoch.png"


# Graph look
python tools/graph.py
