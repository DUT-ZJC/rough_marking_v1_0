"""Shared data structures for STEP-guided STL fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import open3d as o3d


@dataclass
class StepAnalyticFace:
    """One analytic face extracted directly from the STEP B-Rep."""
    id: int
    surface_type: str
    area_mm2: float
    mesh: o3d.geometry.TriangleMesh
    params: dict[str, Any]
    bbox_min: np.ndarray
    bbox_max: np.ndarray


@dataclass
class TransformedStepFace:
    """A STEP analytic face transformed into the STL coordinate system."""
    face: StepAnalyticFace
    mesh: o3d.geometry.TriangleMesh
    params: dict[str, Any]
    bbox_min: np.ndarray
    bbox_max: np.ndarray


@dataclass
class TriangleCache:
    """Cached STL triangle geometry used by fitting and picking."""
    mesh: o3d.geometry.TriangleMesh
    vertices: np.ndarray
    triangles: np.ndarray
    tri_centers: np.ndarray
    tri_normals: np.ndarray
    tri_areas: np.ndarray
    tri_neighbors: list[np.ndarray]


@dataclass
class FitThresholds:
    """User-facing thresholds for support search and residual classification."""
    support_gap_mm: float = 2.0
    plane_tol_mm: float = 1.0
    cylinder_tol_mm: float = 1.0
    generic_tol_mm: float = 1.5
    min_support_triangles: int = 20

    def tolerance_for(self, surface_type: str) -> float:
        """Return the active residual tolerance for the requested surface type."""
        if surface_type == "plane":
            return float(self.plane_tol_mm)
        if surface_type == "cylinder":
            return float(self.cylinder_tol_mm)
        return float(self.generic_tol_mm)


@dataclass
class FaceFitResult:
    """Fitting and classification result for one transformed STEP face."""
    face_id: int
    surface_type: str
    status: str
    message: str
    transformed_face_mesh: o3d.geometry.TriangleMesh
    support_mesh: o3d.geometry.TriangleMesh
    inlier_mesh: o3d.geometry.TriangleMesh
    outlier_mesh: o3d.geometry.TriangleMesh
    support_triangles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    support_residuals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    support_distances: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    inlier_triangles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    outlier_triangles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    fitted_params: dict[str, Any] = field(default_factory=dict)
    support_area_mm2: float = 0.0
    inlier_area_mm2: float = 0.0
    outlier_area_mm2: float = 0.0
    residual_mean_mm: float = 0.0
    residual_p95_mm: float = 0.0
    residual_max_mm: float = 0.0
    inlier_ratio: float = 0.0


@dataclass
class RegistrationConfig:
    """Configuration passed to the coarse STEP-to-STL registration stage."""
    VOXEL_SIZE: float = 2.0
    NORMAL_RADIUS: float = 6.0
    COARSE_ENCLOSE_ENABLE: bool = True
    COARSE_ENCLOSE_MARGIN: float = 0.0
    COARSE_ENCLOSE_SAMPLE_POINTS: int = 12000
    COARSE_ENCLOSE_MAX_ITERS: int = 8

    def as_namespace(self) -> SimpleNamespace:
        """Convert the dataclass into the namespace expected by legacy code."""
        return SimpleNamespace(**self.__dict__)
