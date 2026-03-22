from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from .logging_utils import log

@dataclass
class PlaneFeature:
    normal: np.ndarray  # (3,)
    d: float            # plane: n·x + d = 0
    inlier_indices: np.ndarray  # (M,) indices
    centroid: np.ndarray # (3,)
    area_proxy: float    # proxy from inliers (not exact area)

def _plane_from_model(model: list[float]) -> tuple[np.ndarray, float]:
    # model: [a,b,c,d] for ax+by+cz+d=0
    n = np.array(model[:3], dtype=np.float64)
    d = float(model[3])
    n_norm = np.linalg.norm(n) + 1e-12
    n = n / n_norm
    d = d / n_norm
    return n, d

def detect_planes_iterative(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    max_planes: int = 6,
    min_inliers: int = 800
) -> list[PlaneFeature]:
    """Iteratively segment planes using Open3D RANSAC."""
    log(f"Detecting planes (th={distance_threshold}, max={max_planes})")
    planes: list[PlaneFeature] = []
    work = pcd
    all_points = np.asarray(work.points)

    used_global = np.zeros(len(all_points), dtype=bool)

    # We keep a dynamic pcd for segmentation. To map indices back, we track mask.
    global_idx = np.arange(len(all_points))

    for k in range(max_planes):
        if len(global_idx) < min_inliers:
            break
        model, inliers = work.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)
        if len(inliers) < min_inliers:
            break
        n, d = _plane_from_model(model)
        inliers = np.asarray(inliers, dtype=np.int64)
        g_inliers = global_idx[inliers]

        pts = all_points[g_inliers]
        centroid = pts.mean(axis=0)
        area_proxy = float(len(g_inliers))

        planes.append(PlaneFeature(normal=n, d=d, inlier_indices=g_inliers, centroid=centroid, area_proxy=area_proxy))

        # remove inliers
        mask_keep = np.ones(len(global_idx), dtype=bool)
        mask_keep[inliers] = False
        global_idx = global_idx[mask_keep]
        work = work.select_by_index(inliers.tolist(), invert=True)

        log(f"  plane#{k}: inliers={len(g_inliers)}, n={n}, d={d:.4f}")

    # Sort by inlier count desc
    planes.sort(key=lambda x: x.area_proxy, reverse=True)
    return planes

# Cylinder detection scaffold (TODO)
@dataclass
class CylinderFeature:
    axis_point: np.ndarray
    axis_dir: np.ndarray
    radius: float
    inlier_indices: np.ndarray

def detect_cylinders_stub(*args, **kwargs) -> list[CylinderFeature]:
    log("Cylinder detection is TODO in v1.0 (stub). Returning [].")
    return []
