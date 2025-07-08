"""
Plot a single random point cloud from either partial or complete
"""

import os, random, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 



CAT_FILE = "/home/DAVIDSON/hayavuzkara/Data/22Mg_16O_combo/category.json"
with open(CAT_FILE) as f:
    cats = json.load(f)

train_hashes = cats[0]["train"] + cats[1]["train"]
h = random.choice(train_hashes)           # random hash string


ROOT = "/home/DAVIDSON/hayavuzkara/Data/22Mg_16O_combo"
subset = "train"

partial_path  = os.path.join(ROOT, subset, "partial",  h, "rand.npy")  # or center/down
complete_path = os.path.join(ROOT, subset, "complete", f"{h}.npy")

partial  = np.load(partial_path)         # (≈2048, 3) after transform
complete = np.load(complete_path)        # (51, 3) raw
print(partial.shape)
print(complete.shape)
print("")
print(partial)
print(complete)



fig = plt.figure(figsize=(10,4))


ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(partial[:,0], partial[:,1], partial[:,2], s=1)
ax1.set_title("partial • " + h)


"""
ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(complete[:,0], complete[:,1], complete[:,2], s=1, c="orange")
ax2.set_title("complete • " + h)
"""

plt.tight_layout()
plt.savefig("sample_pc.png", dpi=300)