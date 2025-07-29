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


# To choose random partial point cloud and its gt:
DATA_ROOT="/home/DAVIDSON/hayavuzkara/Data/22Mg"



SPLIT_ROOT="${DATA_ROOT}/test" # Can change to train or val as well. I am using test for it to work with unseen data.
COMPLETE_DIR="${SPLIT_ROOT}/complete"
PARTIAL_DIR="${SPLIT_ROOT}/partial"


# Get a random gt here:
RAND_GT_PATH="$(find "${COMPLETE_DIR}" | shuf -n 1)"
RAND_EVENT_NAME="$(basename "${RAND_GT_PATH}" .npy)"

# Get its random partial cloud here:
RAND_PARTIAL_PATH="${PARTIAL_DIR}/${RAND_EVENT_NAME}/center.npy" #Can be down.npy or center.npy or rand.npy


# Inference

python tools/inference.py cfgs/ALPHA_ATTPC/ALPHA.yaml \
  experiments/ALPHA/ALPHA_ATTPC/ATTPC/MG22Only/ckpt-best.pth \
  --pc "${RAND_PARTIAL_PATH}" --out_pc_root "inference_result/Example_pc" --pc_name "${RAND_EVENT_NAME}"



# Loss vs Epoch plot
EXPERIMENT_PATH="experiments/ALPHA/ALPHA_ATTPC/ATTPC"


# Chooses latest done log
LOG_FILE=$(ls -1 "${EXPERIMENT_PATH}"/[0-9]*.log | sort -t '_' -k1,1 -k2,2 | tail -n 1)


python tools/loss_vs_epoch.py \
  --log_file "${LOG_FILE}" \
  --output "inference_result/loss_vs_epoch.png"

# Graph 
python tools/graph.py \
  --partial_path "${RAND_PARTIAL_PATH}" \
  --gt_path "${RAND_GT_PATH}" \
  --predict_path "inference_result/Example_pc/${RAND_EVENT_NAME}_fine.npy"