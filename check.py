import torch, sys, os
from extensions.chamfer_dist import ChamferDistanceL2

print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

cd = ChamferDistanceL2()
a = torch.rand(1,2048,3,device='cuda'); b = torch.rand(1,16384,3,device='cuda')
dist = cd(a,b); print(dist.device)