"""
Author: Hakan Bora Yavuzkara
Date Created: 07/10/2025

Used to send only 3 dimensions to Distance related functions
"""

from models.Transformer_utils import square_distance
from models.Transformer_utils import knn_point
from pointnet2_ops import pointnet2_utils

def xyz(t):
    # Get first three channels
    return t[..., :3]

def square_distance(src, tgt):
    return square_distance(xyz(src), xyz(tgt))

def knn_point(k, src, query):
    return knn_point(k, xyz(src), xyz(query))

def furthest_point_sample(xyz_tensor, npoints):
    return pointnet2_utils.furthest_point_sample(xyz(xyz_tensor), npoints)
