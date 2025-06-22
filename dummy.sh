#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "dummy"               # name
#SBATCH --output "dummy-out.log"      # output file
#SBATCH --error "dummy-err.log"       # error message file


# resource request info 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu
#SBATCH --constraint cuda11

source /opt/conda/bin/activate env1
which nvcc            # …/envs/env1/bin/nvcc
nvcc --version        # release 11.3
python - <<'PY'
import torch, subprocess, os
print("PyTorch  CUDA:", torch.version.cuda)               # → 11.3
print("nvcc    CUDA:", subprocess.check_output(
      ["nvcc","--version"]).decode().split("release")[1].split(",")[0])
PY