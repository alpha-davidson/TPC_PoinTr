#!/bin/bash

### Job Parameters:
# basic info
#SBATCH --job-name "install"               # name
#SBATCH --output "install-out.log"      # output file
#SBATCH --error "install-err.log"       # error message file

# resource request info
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint cuda11 
#SBATCH --exclude alpha[0-2]

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu


module purge
module load CUDA/11.6.0

source /opt/conda/etc/profile.d/conda.sh

conda activate env1

export TORCH_CUDA_ARCH_LIST="8.6"


HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install

# NOTE: For GRNet 

# Cubic Feature Sampling
cd $HOME/extensions/cubic_feature_sampling
python setup.py install

# Earth Mover Distance
cd $HOME/extensions/emd
python setup.py install

# Gridding & Gridding Reverse
cd $HOME/extensions/gridding
python setup.py install

# Gridding Loss
cd $HOME/extensions/gridding_loss
python setup.py install

# PointNet2
cd $HOME/extensions/Pointnet2_PyTorch/pointnet2_ops_lib/
python setup.py install