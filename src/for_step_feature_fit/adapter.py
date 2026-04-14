"""Compatibility adapter for feeding STEP-guided scan features into the main app.

The main program expects scan-side plane/cylinder features in a shared
application format. This adapter runs the STEP-guided fitting pipeline, keeps
only plane and cylinder results, and converts them into the structures already
consumed by the picker and optimizer.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import open3d as o3d

from ..logging_utils import log
from ..scan_feature_types import ScanCylinderFeature, ScanPlaneFeature
from .core_types import FitThresholds, RegistrationConfig, FaceFitResult
from .pipeline import StepSTLFitSession


def _inlier_residuals(result: FaceFitResult) -> np.ndarray:
    """Extract residual values aligned with the current inlier triangle set."""
    if result.support_triangles.size == 0 or result.inlier_triangles.size == 0:
        return np.zeros(0, dtype=np.float64)
    inlier_set = set(int(tri_id) for tri_id in result.inlier_triangles.tolist())
    mask = np.fromiter(
        (int(tri_id) in inlier_set for tri_id in result.support_triangles.tolist()),
        dtype=bool,
        count=len(result.support_triangles),
    )
    return np.asarray(result.support_residuals[mask], dtype=np.float64)


def _weighted_centroid(
    tri_indices: np.ndarray,
    tri_centers: np.ndarray,
    tri_areas: np.ndarray,
) -> np.ndarray:
    """Compute an area-weighted centroid from STL triangle centers."""
    tri_indices = np.asarray(tri_indices, dtype=np.int32).reshape(-1)
    if tri_indices.size == 0:
        return np.zeros(3, dtype=np.float64)
    weights = np.maximum(tri_areas[tri_indices], 1e-6)
    return np.average(tri_centers[tri_indices], axis=0, weights=weights).astype(np.float64)


def _plane_feature_from_result(
    result: FaceFitResult,
    tri_centers: np.ndarray,
    tri_areas: np.ndarray,
) -> ScanPlaneFeature:
    """Convert one fitted plane result to the legacy scan-plane feature type."""
    params = result.fitted_params
    residuals = _inlier_residuals(result)
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    centroid = _weighted_centroid(result.inlier_triangles, tri_centers, tri_areas)
    return ScanPlaneFeature(
        id=int(result.face_id),
        tri_indices=np.asarray(result.inlier_triangles, dtype=np.int32),
        mesh=o3d.geometry.TriangleMesh(result.inlier_mesh),
        normal=np.asarray(params["normal"], dtype=np.float64),
        d=float(params["d"]),
        centroid=centroid,
        area=float(result.inlier_area_mm2),
        rmse=rmse,
    )


def _cylinder_feature_from_result(
    result: FaceFitResult,
) -> ScanCylinderFeature:
    """Convert one fitted cylinder result to the legacy scan-cylinder feature type."""
    params = result.fitted_params
    residuals = _inlier_residuals(result)
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    return ScanCylinderFeature(
        id=int(result.face_id),
        tri_indices=np.asarray(result.inlier_triangles, dtype=np.int32),
        mesh=o3d.geometry.TriangleMesh(result.inlier_mesh),
        axis_origin=np.asarray(params["axis_origin"], dtype=np.float64),
        axis_dir=np.asarray(params["axis_dir"], dtype=np.float64),
        radius=float(params["radius"]),
        rmse=rmse,
    )


def _build_remaining_mask(
    n_triangles: int,
    planes: Iterable[ScanPlaneFeature],
    cylinders: Iterable[ScanCylinderFeature],
) -> np.ndarray:
    """Return a mask where True means the scan triangle was not assigned to plane/cylinder."""
    mask = np.ones(int(n_triangles), dtype=bool)
    for feat in list(planes) + list(cylinders):
        tri_indices = np.asarray(feat.tri_indices, dtype=np.int32).reshape(-1)
        tri_indices = tri_indices[(tri_indices >= 0) & (tri_indices < n_triangles)]
        mask[tri_indices] = False
    return mask


def process_scan_features_step_guided(
    step_path: str,
    scan_stl_path: str,
    *,
    linear_deflection: float = 0.5,
    thresholds: FitThresholds | None = None,
    registration_config: RegistrationConfig | None = None,
    use_global_results: bool = True,
) -> tuple[list[ScanPlaneFeature], list[ScanCylinderFeature], np.ndarray, o3d.geometry.TriangleMesh]:
    """Extract scan-side plane/cylinder features using the STEP-guided fitter.

    Parameters
    ----------
    step_path:
        CAD STEP file used to localize scan-side analytic features.
    scan_stl_path:
        Scanned STL mesh.
    linear_deflection:
        STEP tessellation deflection.
    thresholds:
        Support search and residual thresholds. Defaults to the module defaults.
    registration_config:
        Configuration forwarded to the coarse registration routine.
    use_global_results:
        When True, run `analyze_all_faces()` so boundary ownership is resolved
        globally before converting features for the main program.
    """
    thresholds = thresholds or FitThresholds()
    registration_config = registration_config or RegistrationConfig()

    session = StepSTLFitSession(
        step_path=step_path,
        scan_stl_path=scan_stl_path,
        linear_deflection=float(linear_deflection),
        registration_config=registration_config,
    )
    session.load()

    results = (
        session.analyze_all_faces(thresholds)
        if bool(use_global_results)
        else [session.analyze_face(face_index, thresholds) for face_index in range(len(session.transformed_faces))]
    )

    cache = session.scan_cache
    assert cache is not None

    planes: list[ScanPlaneFeature] = []
    cylinders: list[ScanCylinderFeature] = []
    for result in results:
        if result.status != "ok" or result.inlier_triangles.size == 0:
            continue
        if result.surface_type == "plane":
            planes.append(_plane_feature_from_result(result, cache.tri_centers, cache.tri_areas))
        elif result.surface_type == "cylinder":
            cylinders.append(_cylinder_feature_from_result(result))

    planes.sort(key=lambda feat: (-float(feat.area), int(feat.id)))
    cylinders.sort(key=lambda feat: (-len(np.asarray(feat.tri_indices)), int(feat.id)))

    remaining_mask = _build_remaining_mask(len(cache.triangles), planes, cylinders)
    log(
        "[STEP-guided] scan features ready: "
        f"planes={len(planes)} cylinders={len(cylinders)} "
        f"remaining={int(remaining_mask.sum())}"
    )
    return planes, cylinders, remaining_mask, o3d.geometry.TriangleMesh(session.scan_mesh)
