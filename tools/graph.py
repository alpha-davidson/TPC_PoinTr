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

RANGES = {
        'MIN_X' : -270.0,
        'MAX_X' :  270.0,
        'MIN_Y' : -270.0,
        'MAX_Y' :  270.0,
        'MIN_Z' : -185.0,
        'MAX_Z' : 1185.0,
        'MIN_LNQ' :  1.0,
        'MAX_LNQ' : 10.2}

def unscale_data(scaled_event, ranges):
    """
    Reverse Min-Max scaling to original coordinate space.
    """
    unscaled = np.zeros_like(scaled_event)

    unscaled[:, 0] = scaled_event[:, 0] * (ranges['MAX_X'] - ranges['MIN_X']) + ranges['MIN_X']
    unscaled[:, 1] = scaled_event[:, 1] * (ranges['MAX_Y'] - ranges['MIN_Y']) + ranges['MIN_Y']
    unscaled[:, 2] = scaled_event[:, 2] * (ranges['MAX_Z'] - ranges['MIN_Z']) + ranges['MIN_Z']
    # unscaled[:, 3] = np.exp(scaled_event[:, 3] * (ranges['MAX_LNQ'] - ranges['MIN_LNQ']) + ranges['MIN_LNQ'])
    return unscaled

BOX_ASPECT=[1,2,1]

# Partial
partial_unscaled=unscale_data(partial,RANGES)
fig1 = plt.figure(figsize=(10,4))
ax1 = plt.axes(projection="3d")
ax1.scatter(partial_unscaled[:, 0], partial_unscaled[:, 1], partial_unscaled[:, 2], s=1)
ax1.set_xlim(-255,255)
ax1.set_ylim(-255,255)
ax1.set_zlim(0,255*4)
ax1.set_xticks(np.arange(-200, 201, 100))
ax1.set_yticks(np.arange(-200, 201, 100))
ax1.set_zticks(np.arange(0, 255*4, 200))
ax1.set_box_aspect(BOX_ASPECT)
ax1.set_title("Partial")
ax1.view_init(elev=20, azim=30, vertical_axis='x') 
plt.savefig(os.path.join(out_dir, "partial.png"), dpi=300)
plt.close()

# Predict
predict_unscaled=unscale_data(predict,RANGES)
fig2 = plt.figure(figsize=(10,4))
ax2 = plt.axes(projection="3d")
ax2.scatter(predict_unscaled[:, 0], predict_unscaled[:, 1], predict_unscaled[:, 2], s=1)
ax2.set_xlim(-255,255)
ax2.set_ylim(-255,255)
ax2.set_zlim(0,255*4)
ax2.set_xticks(np.arange(-200, 201, 100))
ax2.set_yticks(np.arange(-200, 201, 100))
ax2.set_zticks(np.arange(0, 255*4, 200))
ax2.set_box_aspect(BOX_ASPECT)
ax2.set_title("Predict")
ax2.view_init(elev=20, azim=30, vertical_axis='x') 
plt.savefig(os.path.join(out_dir, "predict.png"), dpi=300)
plt.close()

# Ground Truth
if gt_path:
    gt = np.load(gt_path)
    gt_unscaled=unscale_data(gt,RANGES)
    fig3 = plt.figure(figsize=(10,4))
    ax3 = plt.axes(projection="3d")
    ax3.scatter(gt_unscaled[:, 0], gt_unscaled[:, 1], gt_unscaled[:, 2], s=1)
    ax3.set_xlim(-255,255)
    ax3.set_ylim(-255,255)
    ax3.set_zlim(0,255*4)
    ax3.set_xticks(np.arange(-200, 201, 100))
    ax3.set_yticks(np.arange(-200, 201, 100))
    ax3.set_zticks(np.arange(0, 255*4, 200))
    ax3.set_box_aspect(BOX_ASPECT)
    ax3.set_title("GT")
    ax3.view_init(elev=20, azim=30, vertical_axis='x') 
    plt.savefig(os.path.join(out_dir, "GroundTruth.png"), dpi=300)
    plt.close()
