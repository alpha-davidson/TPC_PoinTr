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
import torch, pointnet2_ops, chamfer, emd
print("CUDA available ->", torch.cuda.is_available())
print("Compiled arch   ->", torch.cuda.get_device_capability())
print("PointNet2 ops   ->", pointnet2_ops.__version__)
print("Chamfer OK?     ->", hasattr(chamfer, 'chamfer_distance'))
print("EMD OK?         ->", hasattr(emd, 'emd'))
PY