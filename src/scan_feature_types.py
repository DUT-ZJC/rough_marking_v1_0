from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class ScanPlaneFeature:
    id: int
    tri_indices: np.ndarray
    mesh: o3d.geometry.TriangleMesh
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    area: float
    rmse: float


@dataclass
class ScanCylinderFeature:
    id: int
    tri_indices: np.ndarray
    mesh: o3d.geometry.TriangleMesh
    axis_origin: np.ndarray
    axis_dir: np.ndarray
    radius: float
    rmse: float
