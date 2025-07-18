"""
Plot of loss vs epoch
Date Created: July 2, 2025
Author: Hakan Bora Yavuzkara
"""

import argparse
import os
import re
import matplotlib.pyplot as plt

# To Read log and grab arguments:
p=argparse.ArgumentParser()
p.add_argument("--log_file", required=True)
p.add_argument("--output",   required=True)
args=p.parse_args()

# Detect number in line
float_finder = re.compile(r"[-+]?\d*\.\d+|\d+")

# Access losses (Chamfer Distance Dense Loss)
train_losses=[]
val_losses=[]
loss_re    = re.compile(r"Losses\s*=\s*\[([^\]]+)\]")
metric_re  = re.compile(r"Metrics\s*=\s*\[([^\]]+)\]")

#FIX FIX FIX
with open(args.log_file, "r", encoding="utf-8") as file:
    for line in file:
        # Training loss
        if "[Training]" in line and "Losses =" in line and "EPOCH:" in line:
            m = loss_re.search(line)
            if m:
                first, *_ = float_finder.findall(m.group(1))
                train_losses.append(float(first))
        # Val loss
        elif "[Validation]" in line and "EPOCH:" in line and "Metrics =" in line:
            m = metric_re.search(line)
            if m:
                _fscore, cdl1, *_ = float_finder.findall(m.group(1))
                val_losses.append(float(cdl1))


# For debugging purposes:
if not train_losses:
    raise RuntimeError("No training losses captured – check log format.")
if not val_losses:
    raise RuntimeError("No validation losses captured – check log format.")


# UGLY HERE, FIX 
common_len = min(len(train_losses), len(val_losses))
train_losses = train_losses[:common_len]
val_losses   = val_losses[:common_len]


#Use something better than 1
epochs=range(1,1+common_len)

print(train_losses)
print("")
print(val_losses)

#Plot
plt.figure()
plt.plot(epochs,train_losses,label="Train CD DenseLoss")
plt.plot(epochs,val_losses,label="Val CD DenseLoss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
plt.savefig(args.output, dpi=300)
