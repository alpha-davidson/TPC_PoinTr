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
python - <<'PY'
import torch, re, sys
prop = torch.cuda.get_device_properties(0)
print(f"CUDA capability: {prop.major}.{prop.minor}  ->  sm_{prop.major}{prop.minor}")
PY