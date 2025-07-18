"""
Plot a single random point cloud from partial, gt, and model prediction.
Author: Hakan Bora Yavuzkara
"""

import os, random, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--partial_path', type=str, required=True,
                    help='partial file path')

parser.add_argument('--predict_path', type=str, required=True,
                    help='predicted file path')

parser.add_argument('--gt_path', type=str, required=True,
                    help='gt file path')
args = parser.parse_args()


partial_path = args.partial_path
predict_path = args.predict_path
gt_path = args.gt_path


partial  = np.load(partial_path)         
predict = np.load(predict_path)       
gt = np.load(gt_path)

print(partial.shape)
print(predict.shape)
print(gt.shape)
print("")
print(partial)
print(predict)
print(gt)



out_dir = "inference_result"
os.makedirs(out_dir, exist_ok=True)


X_MIN, X_MAX = -1,1
Y_MIN, Y_MAX = -1,1
Z_MIN, Z_MAX = 0,4
BOX_ASPECT=[X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN]

# Partial
fig1 = plt.figure(figsize=(10,4))
ax1 = plt.axes(projection="3d")
ax1.scatter(partial[:, 0], partial[:, 1], partial[:, 2], s=1)
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_zlim(0,4)
ax1.set_box_aspect(BOX_ASPECT)
ax1.set_title("Partial")
plt.savefig(os.path.join(out_dir, "partial.png"), dpi=300)
plt.close()

# Predict
fig2 = plt.figure(figsize=(10,4))
ax2 = plt.axes(projection="3d")
ax2.scatter(predict[:, 0], predict[:, 1], predict[:, 2], s=1)
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(0,4)
ax2.set_box_aspect(BOX_ASPECT)
ax2.set_title("Predict")
plt.savefig(os.path.join(out_dir, "predict.png"), dpi=300)
plt.close()

# Ground Truth
fig3 = plt.figure(figsize=(10,4))
ax3 = plt.axes(projection="3d")
ax3.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=1)
ax3.set_xlim(-1,1)
ax3.set_ylim(-1,1)
ax3.set_zlim(0,4)
ax3.set_box_aspect(BOX_ASPECT)
ax3.set_title("GT")
plt.savefig(os.path.join(out_dir, "GroundTruth.png"), dpi=300)
plt.close()

