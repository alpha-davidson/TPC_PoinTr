#!/usr/bin/env python
"""
TPC_PoinTr – installation_check
Significantly Chat-GPT generated, shouldn't be too useful mostly except for a single run of making sure installation is fine.
"""

import importlib, inspect, torch, sys, textwrap

GREEN = "\033[92m"; RED = "\033[91m"; END = "\033[0m"
def ok(msg):  print(f"{GREEN}✓{END} {msg}")
def bad(msg): print(f"{RED}✗{END} {msg}")

def compiled_version():
    v = torch._C._cuda_getCompiledVersion()
    return f"{v//1000}.{(v//10)%100:02d}"          # 11060 → 11.6

print("CUDA available :", torch.cuda.is_available())
print("Torch compiled  :", compiled_version())
print("GPU capability :", torch.cuda.get_device_capability())
print("-"*55)

B, N = 2, 128
p1 = torch.randn(B, N, 3, device="cuda")
p2 = torch.randn(B, N, 3, device="cuda")

def try_call(mod_name, *candidates, args=None, kwargs=None):
    """Import *mod_name* then call the first attribute that exists."""
    args   = []  if args   is None else args
    kwargs = {} if kwargs is None else kwargs
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        bad(f"{mod_name:<22} import failed  – {e}")
        return
    for attr in candidates:
        if hasattr(mod, attr):
            try:
                getattr(mod, attr)(*args, **kwargs)
                ok(f"{mod_name:<22} {attr}")
            except Exception as e:
                bad(f"{mod_name:<22} {attr} runtime – {e}")
            return
    bad(f"{mod_name:<22} no expected symbols: {candidates}")

# )Chamfer
try_call(
    "chamfer",                          # binary module
    "ChamferDistance",                  # python wrapper class
    "chamfer_distance", "distChamfer"   # raw CUDA fns
    , args=[p1, p2]
)

# )EMD
try_call(
    "emd",
    "emd", "earth_mover_distance"
    , args=[p1, p2], kwargs={"eps":1e-3, "iters":5}
)

# )Cubic feature sampling
try_call(
    "cubic_feature_sampling",
    "cubic_feature_sampling"
    , args=[p1]
)

# )Gridding
try_call(
    "gridding",
    "gridding", "gridding_forward",
    args=[p1]
)

# )Gridding-distance
try_call(
    "gridding_distance",
    "gridding_distance",
    args=[p1, p2]
)

# )PointNet2 op (furthest-point-sample is enough)
try:
    from pointnet2_ops import pointnet2_utils as p2u
    _ = p2u.furthest_point_sample(p1, 64)
    ok(f"{'pointnet2_ops':<22} furthest_point_sample")
except Exception as e:
    bad(f"{'pointnet2_ops':<22} {e}")

print("-"*55)
print("Done.")