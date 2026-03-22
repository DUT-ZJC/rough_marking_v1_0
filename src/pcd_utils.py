from __future__ import annotations
import numpy as np
import open3d as o3d
from .logging_utils import log

def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel: float, normal_radius: float) -> o3d.geometry.PointCloud:
    log(f"Downsample voxel={voxel}")
    pcd_ds = pcd.voxel_down_sample(voxel)
    log(f"Estimate normals radius={normal_radius}")
    pcd_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    pcd_ds.orient_normals_consistent_tangent_plane(30)
    return pcd_ds

def to_o3d_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return p
