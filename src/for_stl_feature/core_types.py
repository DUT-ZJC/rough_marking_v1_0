# core_types.py
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from abc import ABC, abstractmethod
from typing import Tuple, List

@dataclass
class ScanPlaneFeature:
    id: int
    tri_indices: np.ndarray          # (K,)
    mesh: o3d.geometry.TriangleMesh  # colored overlay mesh
    normal: np.ndarray               # (3,)
    d: float
    centroid: np.ndarray             # (3,)
    area: float
    rmse: float

@dataclass
class ScanCylinderFeature:
    id: int
    tri_indices: np.ndarray
    mesh: o3d.geometry.TriangleMesh
    axis_origin: np.ndarray         # (3,)
    axis_dir: np.ndarray            # (3,) unit
    radius: float
    rmse: float


class BaseFeatureExtractor(ABC):
    """
    所有特征提取算法的基类模板 (纯虚接口)。
    任何新加入的算法都必须继承此类，并实现 extract 方法。
    """
    @abstractmethod
    def extract(
        self, 
        scan_stl: str, 
        **kwargs
    ) -> Tuple[List[ScanPlaneFeature], List[ScanCylinderFeature], np.ndarray, o3d.geometry.TriangleMesh]:
        pass