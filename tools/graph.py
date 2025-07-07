"""
Plot a single random point cloud from either partial or complete
Author: Hakan Bora Yavuzkara
"""

import os, random, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 



ROOT = "/home/DAVIDSON/hayavuzkara/TPC_PoinTr"

# ROOT = "/home/DAVIDSON/hayavuzkara/TPC_PoinTr/demo/ALPHA"

#CAN CHANGE WHERE THE FILE IS LOCATED OR CALLED
# MAKE IT MORE GLOBAL AND PULL FROM DATAA
partial_path  = os.path.join(ROOT, "demo", "ALPHA", "center.npy")  # or rand/down/center any
complete_path = os.path.join(ROOT, "inference_result", "center", "fine.npy")
gt_path = os.path.join(ROOT, "demo", "ALPHA", "0ec2c19ffb88e3d1b4c6473dac65ce75.npy")

partial  = np.load(partial_path)         
complete = np.load(complete_path)       
gt = np.load(gt_path)
print(partial.shape)
print(complete.shape)
print(gt.shape)
print("")
print(partial)
print(complete)
print(gt)



out_dir = os.path.join(ROOT,"inference_result")
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

# Complete
fig2 = plt.figure(figsize=(10,4))
ax2 = plt.axes(projection="3d")
ax2.scatter(complete[:, 0], complete[:, 1], complete[:, 2], s=1)
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(0,4)
ax2.set_box_aspect(BOX_ASPECT)
ax2.set_title("Complete")
plt.savefig(os.path.join(out_dir, "complete.png"), dpi=300)
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

