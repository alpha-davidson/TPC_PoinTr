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

parser.add_argument('--gt_path', type=str, required=False,
                    help='gt file path')

args = parser.parse_args()


partial_path = args.partial_path
predict_path = args.predict_path
gt_path = args.gt_path


partial  = np.load(partial_path)         
predict = np.load(predict_path)       



out_dir = "inference_result"
os.makedirs(out_dir, exist_ok=True)



BOX_ASPECT=[1,2,1]

# Partial
partial_scaled=partial*255
fig1 = plt.figure(figsize=(10,4))
ax1 = plt.axes(projection="3d")
ax1.scatter(partial_scaled[:, 0], partial_scaled[:, 1], partial_scaled[:, 2], s=1)
ax1.set_xlim(-255,255)
ax1.set_ylim(-255,255)
ax1.set_zlim(0,255*4)
ax1.set_xticks(np.arange(-200, 201, 100))
ax1.set_yticks(np.arange(-200, 201, 100))
ax1.set_zticks(np.arange(0, 255*4, 200))
ax1.set_box_aspect(BOX_ASPECT)
ax1.set_title("Partial")
ax1.view_init(elev=15, azim=15, vertical_axis='x') 
plt.savefig(os.path.join(out_dir, "partial.png"), dpi=300)
plt.close()

# Predict
predict_scaled=predict*255
fig2 = plt.figure(figsize=(10,4))
ax2 = plt.axes(projection="3d")
ax2.scatter(predict_scaled[:, 0], predict_scaled[:, 1], predict_scaled[:, 2], s=1)
ax2.set_xlim(-255,255)
ax2.set_ylim(-255,255)
ax2.set_zlim(0,255*4)
ax2.set_xticks(np.arange(-200, 201, 100))
ax2.set_yticks(np.arange(-200, 201, 100))
ax2.set_zticks(np.arange(0, 255*4, 200))
ax2.set_box_aspect(BOX_ASPECT)
ax2.set_title("Predict")
ax2.view_init(elev=15, azim=15, vertical_axis='x') 
plt.savefig(os.path.join(out_dir, "predict.png"), dpi=300)
plt.close()

# Ground Truth
if gt_path:
    gt = np.load(gt_path)
    gt_scaled=gt*255
    fig3 = plt.figure(figsize=(10,4))
    ax3 = plt.axes(projection="3d")
    ax3.scatter(gt_scaled[:, 0], gt_scaled[:, 1], gt_scaled[:, 2], s=1)
    ax3.set_xlim(-255,255)
    ax3.set_ylim(-255,255)
    ax3.set_zlim(0,255*4)
    ax3.set_xticks(np.arange(-200, 201, 100))
    ax3.set_yticks(np.arange(-200, 201, 100))
    ax3.set_zticks(np.arange(0, 255*4, 200))
    ax3.set_box_aspect(BOX_ASPECT)
    ax3.set_title("GT")
    ax3.view_init(elev=15, azim=15, vertical_axis='x') 
    plt.savefig(os.path.join(out_dir, "GroundTruth.png"), dpi=300)
    plt.close()
