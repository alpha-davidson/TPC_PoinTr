#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "inference"               # name
#SBATCH --output "inference-out.log"      # output file
#SBATCH --error "inference-err.log"       # error message file

# resource request info 
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu


source /opt/conda/bin/activate env1

export TORCH_CUDA_ARCH_LIST="8.6"

python tools/inference.py cfgs/ALPHA_ATTPC/ALPHA.yaml \
  experiments/ALPHA/ALPHA_ATTPC/ATTPC/ckpt-best.pth \
  --pc_root demo/ --save_vis_img --out_pc_root inference_result/