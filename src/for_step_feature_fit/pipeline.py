"""STEP-guided local fitting of analytic surfaces on an STL mesh.

The pipeline uses STEP faces only to localize where a feature should exist on
the scan mesh. Final plane / cylinder / cone / sphere / torus parameters are
all re-fitted from local STL triangles. Residual thresholds are then applied to
separate inliers from burrs or defects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..io_stl import load_stl_mesh
from ..logging_utils import log
from ..registration_global import registration_coarse
from .core_types import (
    FaceFitResult,
    FitThresholds,
    RegistrationConfig,
    StepAnalyticFace,
    TransformedStepFace,
    TriangleCache,
)
from .step_reader import extract_step_analytic_faces


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _rotate_vector_by_rotvec(vector: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return vector.copy()
    axis = rotvec / theta
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return (
        vector * cos_t
        + np.cross(axis, vector) * sin_t
        + axis * float(axis @ vector) * (1.0 - cos_t)
    )


def _orthonormal_frame(axis_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis_dir = _unit(axis_dir.astype(np.float64))
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(ref @ axis_dir)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _unit(np.cross(axis_dir, ref))
    w = _unit(np.cross(axis_dir, u))
    return u, w


def _dominant_triangle_normal(tri_normals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    tri_normals = np.asarray(tri_normals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(tri_normals) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    ref_index = int(np.argmax(weights)) if len(weights) else 0
    ref_normal = _unit(tri_normals[ref_index])
    aligned = tri_normals.copy()
    aligned[(aligned @ ref_normal) < 0.0] *= -1.0
    accum = (aligned * weights[:, None]).sum(axis=0)
    if np.linalg.norm(accum) < 1e-12:
        return ref_normal
    return _unit(accum)


def _estimate_spread_axis(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(points) < 2:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    order = np.argsort(weights)
    sample = points[order[-min(len(points), 48):]]
    if len(sample) < 2:
        sample = points

    best_vec = sample[-1] - sample[0]
    best_norm = float(np.linalg.norm(best_vec))
    for i in range(len(sample) - 1):
        pi = sample[i]
        diff = sample[i + 1:] - pi[None, :]
        norms = np.linalg.norm(diff, axis=1)
        if norms.size == 0:
            continue
        j = int(np.argmax(norms))
        if float(norms[j]) > best_norm:
            best_norm = float(norms[j])
            best_vec = diff[j]

    if best_norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return _unit(best_vec)


def _robust_residual_mask(residual: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    if len(residual) < 8:
        return np.ones(len(residual), dtype=bool)

    median = float(np.median(residual))
    p70 = float(np.percentile(residual, 70))
    p85 = float(np.percentile(residual, 85))
    limit = max(median * 2.5, p70, 1e-4)
    limit = min(limit, p85)
    mask = residual <= limit
    if int(mask.sum()) < max(6, len(residual) // 5):
        return residual <= p85
    return mask


def _submesh_from_triangles(
    mesh: o3d.geometry.TriangleMesh,
    tri_idx: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    tri_idx = np.asarray(tri_idx, dtype=np.int64)
    if tri_idx.size == 0:
        return o3d.geometry.TriangleMesh()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    tri_idx = tri_idx[(tri_idx >= 0) & (tri_idx < len(triangles))]
    if tri_idx.size == 0:
        return o3d.geometry.TriangleMesh()

    sub_triangles = triangles[tri_idx]
    used = np.unique(sub_triangles.reshape(-1))
    if used.size == 0:
        return o3d.geometry.TriangleMesh()

    remap = {int(old): i for i, old in enumerate(used.tolist())}
    remapped = np.vectorize(remap.get)(sub_triangles)
    sub = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices[used].astype(np.float64)),
        o3d.utility.Vector3iVector(remapped.astype(np.int32)),
    )
    if len(sub.triangles) > 0:
        sub.compute_vertex_normals()
    return sub


def _build_triangle_neighbors(triangles: np.ndarray) -> list[np.ndarray]:
    triangles = np.asarray(triangles, dtype=np.int32)
    neighbors: list[set[int]] = [set() for _ in range(len(triangles))]
    edge_to_tris: dict[tuple[int, int], list[int]] = {}

    for tri_id, tri in enumerate(triangles):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            edge_to_tris.setdefault(edge, []).append(tri_id)

    for tri_ids in edge_to_tris.values():
        if len(tri_ids) < 2:
            continue
        for i, tri_i in enumerate(tri_ids[:-1]):
            for tri_j in tri_ids[i + 1:]:
                neighbors[tri_i].add(tri_j)
                neighbors[tri_j].add(tri_i)

    return [np.asarray(sorted(nbrs), dtype=np.int32) for nbrs in neighbors]


def _triangle_areas_and_centers(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    centers = (p0 + p1 + p2) / 3.0
    return area.astype(np.float64), centers.astype(np.float64)


def _build_triangle_cache(mesh: o3d.geometry.TriangleMesh) -> TriangleCache:
    mesh = o3d.geometry.TriangleMesh(mesh)
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    tri_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    tri_areas, tri_centers = _triangle_areas_and_centers(vertices, triangles)
    tri_neighbors = _build_triangle_neighbors(triangles)
    return TriangleCache(
        mesh=mesh,
        vertices=vertices,
        triangles=triangles,
        tri_centers=tri_centers,
        tri_normals=tri_normals,
        tri_areas=tri_areas,
        tri_neighbors=tri_neighbors,
    )


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ R.T + t[None, :]


def _transform_dirs(vectors: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    out = np.asarray(vectors, dtype=np.float64) @ R.T
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def _transform_mesh(mesh: o3d.geometry.TriangleMesh, T: np.ndarray) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(mesh)
    out.transform(np.asarray(T, dtype=np.float64))
    if not out.has_vertex_normals():
        out.compute_vertex_normals()
    return out


def _transform_face(face: StepAnalyticFace, T: np.ndarray) -> TransformedStepFace:
    transformed_mesh = _transform_mesh(face.mesh, T)
    vertices = np.asarray(transformed_mesh.vertices, dtype=np.float64)
    bbox_min = vertices.min(axis=0) if len(vertices) else np.zeros(3, dtype=np.float64)
    bbox_max = vertices.max(axis=0) if len(vertices) else np.zeros(3, dtype=np.float64)

    params = dict(face.params)
    if face.surface_type == "plane":
        point = _transform_points(np.asarray([params["point"]], dtype=np.float64), T)[0]
        normal = _transform_dirs(np.asarray([params["normal"]], dtype=np.float64), T)[0]
        params["point"] = point
        params["normal"] = normal
        params["d"] = -float(normal @ point)
    elif face.surface_type == "cylinder":
        params["axis_origin"] = _transform_points(np.asarray([params["axis_origin"]], dtype=np.float64), T)[0]
        params["axis_dir"] = _transform_dirs(np.asarray([params["axis_dir"]], dtype=np.float64), T)[0]
    elif face.surface_type == "cone":
        params["apex"] = _transform_points(np.asarray([params["apex"]], dtype=np.float64), T)[0]
        params["axis_dir"] = _transform_dirs(np.asarray([params["axis_dir"]], dtype=np.float64), T)[0]
    elif face.surface_type == "sphere":
        params["center"] = _transform_points(np.asarray([params["center"]], dtype=np.float64), T)[0]
    elif face.surface_type == "torus":
        params["center"] = _transform_points(np.asarray([params["center"]], dtype=np.float64), T)[0]
        params["axis_dir"] = _transform_dirs(np.asarray([params["axis_dir"]], dtype=np.float64), T)[0]

    return TransformedStepFace(
        face=face,
        mesh=transformed_mesh,
        params=params,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def _pairwise_axis_from_normals(
    normals: np.ndarray,
    weights: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    if len(normals) == 0:
        return _unit(fallback.astype(np.float64))

    idx = np.linspace(0, len(normals) - 1, min(len(normals), 48)).astype(np.int32)
    ns = normals[idx]
    ws = weights[idx]
    accum = np.zeros(3, dtype=np.float64)
    for i in range(len(ns)):
        ni = _unit(ns[i])
        for j in range(i + 1, len(ns)):
            nj = _unit(ns[j])
            cp = np.cross(ni, nj)
            n_cp = np.linalg.norm(cp)
            if n_cp < 1e-8:
                continue
            cp = cp / n_cp
            w = float(ws[i] * ws[j] * n_cp)
            if np.linalg.norm(accum) > 1e-12 and float(cp @ accum) < 0.0:
                cp = -cp
            accum += w * cp

    if np.linalg.norm(accum) > 1e-12:
        axis = _unit(accum)
    else:
        axis = _unit(fallback.astype(np.float64))

    if float(axis @ fallback) < 0.0:
        axis = -axis
    return axis


def _fit_circle_2d(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if len(points_xy) == 0:
        return np.zeros(2, dtype=np.float64), 0.0
    if len(points_xy) < 3:
        center = points_xy.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(points_xy - center[None, :], axis=1)))
        return center.astype(np.float64), radius

    pts = points_xy[np.linspace(0, len(points_xy) - 1, min(len(points_xy), 24)).astype(np.int32)]
    centers = []
    weights = []
    for i in range(len(pts) - 2):
        p1 = pts[i]
        for j in range(i + 1, len(pts) - 1):
            p2 = pts[j]
            for k in range(j + 1, len(pts)):
                p3 = pts[k]
                det = 2.0 * (
                    p1[0] * (p2[1] - p3[1])
                    + p2[0] * (p3[1] - p1[1])
                    + p3[0] * (p1[1] - p2[1])
                )
                if abs(det) < 1e-8:
                    continue
                s1 = p1[0] * p1[0] + p1[1] * p1[1]
                s2 = p2[0] * p2[0] + p2[1] * p2[1]
                s3 = p3[0] * p3[0] + p3[1] * p3[1]
                cx = (s1 * (p2[1] - p3[1]) + s2 * (p3[1] - p1[1]) + s3 * (p1[1] - p2[1])) / det
                cy = (s1 * (p3[0] - p2[0]) + s2 * (p1[0] - p3[0]) + s3 * (p2[0] - p1[0])) / det
                centers.append([cx, cy])
                weights.append(abs(det))

    if centers:
        center = np.average(np.asarray(centers, dtype=np.float64), axis=0, weights=np.asarray(weights, dtype=np.float64))
    else:
        center = pts.mean(axis=0)

    radius = float(np.mean(np.linalg.norm(points_xy - center[None, :], axis=1)))
    return center.astype(np.float64), radius


def _normalized_weight_scale(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    mean_w = max(float(np.mean(weights)), 1e-12)
    return np.sqrt(np.maximum(weights, 1e-12) / mean_w)


def _empty_result(face: TransformedStepFace, message: str) -> FaceFitResult:
    empty = o3d.geometry.TriangleMesh()
    return FaceFitResult(
        face_id=face.face.id,
        surface_type=face.face.surface_type,
        status="empty",
        message=message,
        transformed_face_mesh=o3d.geometry.TriangleMesh(face.mesh),
        support_mesh=empty,
        inlier_mesh=empty,
        outlier_mesh=empty,
    )


def _fit_plane(
    points: np.ndarray,
    tri_normals: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    tri_normals = np.asarray(tri_normals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    def _fit_plane_geometry(
        fit_points: np.ndarray,
        fit_normals: np.ndarray,
        fit_weights: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        normal = _dominant_triangle_normal(fit_normals, fit_weights)
        point = np.average(fit_points, axis=0, weights=fit_weights)
        residual_signed = (points - point[None, :]) @ normal
        residual = np.abs(residual_signed)
        d = -float(normal @ point)
        return {
            "normal": normal,
            "point": point,
            "d": d,
        }, residual.astype(np.float64)

    init_params, residual = _fit_plane_geometry(points, tri_normals, weights)
    normal0 = init_params["normal"]
    f_scale = max(float(np.percentile(np.abs(residual), 75)), 1e-3)

    def _solve(
        fit_points: np.ndarray,
        fit_normals: np.ndarray,
        fit_weights: np.ndarray,
        seed_normal: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        base_normal = _dominant_triangle_normal(fit_normals, fit_weights)
        if float(base_normal @ seed_normal) < 0.0:
            base_normal = -base_normal
        base_centroid = np.average(fit_points, axis=0, weights=fit_weights)

        def residual_fn(params: np.ndarray) -> np.ndarray:
            rotvec = params[:3]
            offset = float(params[3])
            normal = _unit(_rotate_vector_by_rotvec(base_normal, rotvec))
            point = base_centroid + offset * normal
            signed = (fit_points - point[None, :]) @ normal
            return _normalized_weight_scale(fit_weights) * signed

        init = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = least_squares(
            residual_fn,
            init,
            loss="soft_l1",
            f_scale=max(f_scale, 1e-3),
            max_nfev=200,
        )
        if not result.success:
            return _fit_plane_geometry(fit_points, fit_normals, fit_weights)

        rotvec = result.x[:3]
        offset = float(result.x[3])
        normal = _unit(_rotate_vector_by_rotvec(base_normal, rotvec))
        point = base_centroid + offset * normal
        if float(normal @ seed_normal) < 0.0:
            normal = -normal
        signed_all = (points - point[None, :]) @ normal
        residual_all = np.abs(signed_all)
        d = -float(normal @ point)
        return {
            "normal": normal,
            "point": point,
            "d": d,
        }, residual_all.astype(np.float64)

    fitted_params, residual = _solve(points, tri_normals, weights, normal0)
    refine_mask = _robust_residual_mask(residual)
    if int(refine_mask.sum()) >= 6 and int(refine_mask.sum()) < len(points):
        fitted_params, residual = _solve(
            points[refine_mask],
            tri_normals[refine_mask],
            weights[refine_mask],
            fitted_params["normal"],
        )
    return fitted_params, residual


def _connected_components_from_subset(
    subset_triangles: np.ndarray,
    tri_neighbors: list[np.ndarray],
) -> list[np.ndarray]:
    subset = np.asarray(subset_triangles, dtype=np.int32)
    if subset.size == 0:
        return []

    allowed = set(int(idx) for idx in subset.tolist())
    visited: set[int] = set()
    components: list[np.ndarray] = []

    for start in subset.tolist():
        if start in visited:
            continue
        stack = [int(start)]
        visited.add(int(start))
        comp: list[int] = []

        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nbr in tri_neighbors[cur]:
                nbr_i = int(nbr)
                if nbr_i not in allowed or nbr_i in visited:
                    continue
                visited.add(nbr_i)
                stack.append(nbr_i)

        components.append(np.asarray(comp, dtype=np.int32))

    return components


def _plane_normal_consistency_mask(
    tri_normals: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dominant = _dominant_triangle_normal(tri_normals, weights)
    dots = np.abs(np.asarray(tri_normals, dtype=np.float64) @ dominant)
    if len(dots) == 0:
        return np.zeros(0, dtype=bool), dominant

    p15 = float(np.percentile(dots, 15))
    align_threshold = max(0.78, min(0.985, p15))
    mask = dots >= align_threshold
    if int(mask.sum()) < max(8, len(dots) // 4):
        mask = dots >= max(0.72, float(np.percentile(dots, 8)))
    return mask, dominant


def _purify_plane_support(
    face: TransformedStepFace,
    support_triangles: np.ndarray,
    cache: TriangleCache,
    face_scene: o3d.t.geometry.RaycastingScene,
    min_keep: int,
) -> np.ndarray:
    """Remove obvious neighbor planes before plane fitting."""
    support_triangles = np.asarray(support_triangles, dtype=np.int32)
    if support_triangles.size == 0:
        return support_triangles

    tri_normals = cache.tri_normals[support_triangles]
    weights = np.maximum(cache.tri_areas[support_triangles], 1e-6)
    normal_mask, _ = _plane_normal_consistency_mask(tri_normals, weights)
    filtered = support_triangles[normal_mask].astype(np.int32)
    if filtered.size < max(min_keep, 8):
        return support_triangles
    return _select_best_support_component(filtered, cache, face_scene, min_keep=min_keep)


def _fit_cylinder(
    points: np.ndarray,
    tri_normals: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    tri_normals = np.asarray(tri_normals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    def _fit_cylinder_geometry(
        fit_points: np.ndarray,
        fit_normals: np.ndarray,
        fit_weights: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        axis_hint = _estimate_spread_axis(fit_points, fit_weights)
        axis_dir = _pairwise_axis_from_normals(fit_normals, fit_weights, axis_hint)
        u, w = _orthonormal_frame(axis_dir)

        projected = fit_points - np.outer(fit_points @ axis_dir, axis_dir)
        xy = np.stack([projected @ u, projected @ w], axis=1)
        center_2d, _ = _fit_circle_2d(xy)
        alpha = float(np.average(fit_points @ axis_dir, weights=fit_weights))
        axis_origin = u * center_2d[0] + w * center_2d[1] + axis_dir * alpha

        radial = points - axis_origin[None, :]
        radial = radial - np.outer(radial @ axis_dir, axis_dir)
        radial_norm = np.linalg.norm(radial, axis=1)
        radius = float(np.average(radial_norm, weights=weights))
        residual = np.abs(radial_norm - radius)
        return {
            "axis_origin": axis_origin.astype(np.float64),
            "axis_dir": axis_dir.astype(np.float64),
            "radius": radius,
        }, residual.astype(np.float64)

    init_params, residual = _fit_cylinder_geometry(points, tri_normals, weights)
    axis0 = _unit(init_params["axis_dir"])
    centroid0 = np.average(points, axis=0, weights=weights)
    u0, v0 = _orthonormal_frame(axis0)
    rel0 = init_params["axis_origin"] - centroid0
    offset0_u = float(rel0 @ u0)
    offset0_v = float(rel0 @ v0)
    log_radius0 = float(np.log(max(init_params["radius"], 1e-6)))
    f_scale = max(float(np.percentile(np.abs(residual), 75)), 1e-3)

    def _solve(
        fit_points: np.ndarray,
        fit_normals: np.ndarray,
        fit_weights: np.ndarray,
        seed_axis: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        base_axis_hint = _estimate_spread_axis(fit_points, fit_weights)
        base_axis = _pairwise_axis_from_normals(fit_normals, fit_weights, base_axis_hint)
        if float(base_axis @ seed_axis) < 0.0:
            base_axis = -base_axis
        base_centroid = np.average(fit_points, axis=0, weights=fit_weights)
        base_u, base_v = _orthonormal_frame(base_axis)
        rel = init_params["axis_origin"] - base_centroid
        init = np.array(
            [0.0, 0.0, 0.0, float(rel @ base_u), float(rel @ base_v), log_radius0],
            dtype=np.float64,
        )
        fit_weight_scale = _normalized_weight_scale(fit_weights)

        def residual_fn(params: np.ndarray) -> np.ndarray:
            rotvec = params[:3]
            axis_dir = _unit(_rotate_vector_by_rotvec(base_axis, rotvec))
            u, v = _orthonormal_frame(axis_dir)
            axis_origin = base_centroid + float(params[3]) * u + float(params[4]) * v
            radius = float(np.exp(params[5]))

            radial = fit_points - axis_origin[None, :]
            radial = radial - np.outer(radial @ axis_dir, axis_dir)
            radial_norm = np.linalg.norm(radial, axis=1)
            return fit_weight_scale * (radial_norm - radius)

        result = least_squares(
            residual_fn,
            init,
            loss="soft_l1",
            f_scale=max(f_scale, 1e-3),
            max_nfev=300,
        )
        if not result.success:
            return _fit_cylinder_geometry(fit_points, fit_normals, fit_weights)

        rotvec = result.x[:3]
        axis_dir = _unit(_rotate_vector_by_rotvec(base_axis, rotvec))
        if float(axis_dir @ seed_axis) < 0.0:
            axis_dir = -axis_dir
        u, v = _orthonormal_frame(axis_dir)
        axis_origin = base_centroid + float(result.x[3]) * u + float(result.x[4]) * v
        radius = float(np.exp(result.x[5]))

        radial = points - axis_origin[None, :]
        radial = radial - np.outer(radial @ axis_dir, axis_dir)
        radial_norm = np.linalg.norm(radial, axis=1)
        residual_all = np.abs(radial_norm - radius)
        return {
            "axis_origin": axis_origin.astype(np.float64),
            "axis_dir": axis_dir.astype(np.float64),
            "radius": radius,
        }, residual_all.astype(np.float64)

    fitted_params, residual = _solve(points, tri_normals, weights, axis0)
    refine_mask = _robust_residual_mask(residual)
    if int(refine_mask.sum()) >= 8 and int(refine_mask.sum()) < len(points):
        fitted_params, residual = _solve(
            points[refine_mask],
            tri_normals[refine_mask],
            weights[refine_mask],
            fitted_params["axis_dir"],
        )
    return fitted_params, residual


def _fit_sphere(
    points: np.ndarray,
    weights: np.ndarray,
    init_params: dict,
) -> tuple[dict, np.ndarray]:
    """Fit a sphere to STL triangle centers with robust least squares."""
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    init_center = np.asarray(init_params["center"], dtype=np.float64)
    init_radius = max(float(init_params["radius"]), 1e-6)

    def _fit_sphere_geometry(
        fit_points: np.ndarray,
        fit_weights: np.ndarray,
        center_guess: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        radial = np.linalg.norm(points - center_guess[None, :], axis=1)
        radius = float(np.average(np.linalg.norm(fit_points - center_guess[None, :], axis=1), weights=fit_weights))
        residual = np.abs(radial - radius)
        return {
            "center": center_guess.astype(np.float64),
            "radius": radius,
        }, residual.astype(np.float64)

    init_fit, init_residual = _fit_sphere_geometry(points, weights, init_center)
    f_scale = max(float(np.percentile(np.abs(init_residual), 75)), 1e-3)

    def _solve(
        fit_points: np.ndarray,
        fit_weights: np.ndarray,
        seed_center: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        fit_weight_scale = _normalized_weight_scale(fit_weights)
        seed_radius = max(
            float(np.average(np.linalg.norm(fit_points - seed_center[None, :], axis=1), weights=fit_weights)),
            init_radius,
            1e-6,
        )

        def residual_fn(params: np.ndarray) -> np.ndarray:
            center = seed_center + params[:3]
            radius = float(np.exp(params[3]))
            radial = np.linalg.norm(fit_points - center[None, :], axis=1)
            return fit_weight_scale * (radial - radius)

        result = least_squares(
            residual_fn,
            np.array([0.0, 0.0, 0.0, float(np.log(seed_radius))], dtype=np.float64),
            loss="soft_l1",
            f_scale=f_scale,
            max_nfev=250,
        )
        if not result.success:
            return _fit_sphere_geometry(fit_points, fit_weights, seed_center)

        center = seed_center + result.x[:3]
        radius = float(np.exp(result.x[3]))
        residual_all = np.abs(np.linalg.norm(points - center[None, :], axis=1) - radius)
        return {
            "center": center.astype(np.float64),
            "radius": radius,
        }, residual_all.astype(np.float64)

    fitted_params, residual = _solve(points, weights, init_fit["center"])
    refine_mask = _robust_residual_mask(residual)
    if int(refine_mask.sum()) >= 8 and int(refine_mask.sum()) < len(points):
        fitted_params, residual = _solve(
            points[refine_mask],
            weights[refine_mask],
            fitted_params["center"],
        )
    return fitted_params, residual


def _fit_cone(
    points: np.ndarray,
    weights: np.ndarray,
    init_params: dict,
) -> tuple[dict, np.ndarray]:
    """Fit a cone to STL triangle centers with robust least squares."""
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    init_apex = np.asarray(init_params["apex"], dtype=np.float64)
    init_axis = _unit(np.asarray(init_params["axis_dir"], dtype=np.float64))
    init_angle = float(init_params["semi_angle_rad"])
    init_tan = max(float(np.tan(np.clip(init_angle, 1e-4, np.pi / 2.0 - 1e-4))), 1e-6)

    def _cone_residual(point_set: np.ndarray, apex: np.ndarray, axis_dir: np.ndarray, tan_angle: float) -> np.ndarray:
        vec = point_set - apex[None, :]
        axial = vec @ axis_dir
        radial = np.linalg.norm(vec - np.outer(axial, axis_dir), axis=1)
        return radial - np.abs(axial) * tan_angle

    init_residual = np.abs(_cone_residual(points, init_apex, init_axis, init_tan))
    f_scale = max(float(np.percentile(init_residual, 75)), 1e-3)

    def _solve(
        fit_points: np.ndarray,
        fit_weights: np.ndarray,
        seed_apex: np.ndarray,
        seed_axis: np.ndarray,
        seed_tan: float,
    ) -> tuple[dict, np.ndarray]:
        fit_weight_scale = _normalized_weight_scale(fit_weights)

        def residual_fn(params: np.ndarray) -> np.ndarray:
            apex = seed_apex + params[:3]
            axis_dir = _unit(_rotate_vector_by_rotvec(seed_axis, params[3:6]))
            tan_angle = float(np.exp(params[6]))
            residual_signed = _cone_residual(fit_points, apex, axis_dir, tan_angle)
            return fit_weight_scale * residual_signed

        result = least_squares(
            residual_fn,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(np.log(max(seed_tan, 1e-6)))], dtype=np.float64),
            loss="soft_l1",
            f_scale=f_scale,
            max_nfev=320,
        )
        if not result.success:
            residual_all = np.abs(_cone_residual(points, seed_apex, seed_axis, seed_tan))
            return {
                "apex": seed_apex.astype(np.float64),
                "axis_dir": seed_axis.astype(np.float64),
                "semi_angle_rad": float(np.arctan(seed_tan)),
                "ref_radius": float(init_params.get("ref_radius", 0.0)),
            }, residual_all.astype(np.float64)

        apex = seed_apex + result.x[:3]
        axis_dir = _unit(_rotate_vector_by_rotvec(seed_axis, result.x[3:6]))
        tan_angle = float(np.exp(result.x[6]))
        residual_all = np.abs(_cone_residual(points, apex, axis_dir, tan_angle))
        return {
            "apex": apex.astype(np.float64),
            "axis_dir": axis_dir.astype(np.float64),
            "semi_angle_rad": float(np.arctan(tan_angle)),
            "ref_radius": float(init_params.get("ref_radius", 0.0)),
        }, residual_all.astype(np.float64)

    fitted_params, residual = _solve(points, weights, init_apex, init_axis, init_tan)
    refine_mask = _robust_residual_mask(residual)
    if int(refine_mask.sum()) >= 10 and int(refine_mask.sum()) < len(points):
        fitted_params, residual = _solve(
            points[refine_mask],
            weights[refine_mask],
            np.asarray(fitted_params["apex"], dtype=np.float64),
            _unit(np.asarray(fitted_params["axis_dir"], dtype=np.float64)),
            max(float(np.tan(fitted_params["semi_angle_rad"])), 1e-6),
        )
    return fitted_params, residual


def _fit_torus(
    points: np.ndarray,
    weights: np.ndarray,
    init_params: dict,
) -> tuple[dict, np.ndarray]:
    """Fit a torus to STL triangle centers with robust least squares."""
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    init_center = np.asarray(init_params["center"], dtype=np.float64)
    init_axis = _unit(np.asarray(init_params["axis_dir"], dtype=np.float64))
    init_minor = max(float(init_params["minor_radius"]), 1e-6)
    init_major = max(float(init_params["major_radius"]), init_minor + 1e-6)
    init_gap = max(init_major - init_minor, 1e-6)

    def _torus_residual(
        point_set: np.ndarray,
        center: np.ndarray,
        axis_dir: np.ndarray,
        major_radius: float,
        minor_radius: float,
    ) -> np.ndarray:
        vec = point_set - center[None, :]
        axial = vec @ axis_dir
        radial = np.linalg.norm(vec - np.outer(axial, axis_dir), axis=1)
        tube = np.sqrt((radial - major_radius) ** 2 + axial ** 2)
        return tube - minor_radius

    init_residual = np.abs(_torus_residual(points, init_center, init_axis, init_major, init_minor))
    f_scale = max(float(np.percentile(init_residual, 75)), 1e-3)

    def _solve(
        fit_points: np.ndarray,
        fit_weights: np.ndarray,
        seed_center: np.ndarray,
        seed_axis: np.ndarray,
        seed_major: float,
        seed_minor: float,
    ) -> tuple[dict, np.ndarray]:
        fit_weight_scale = _normalized_weight_scale(fit_weights)
        seed_minor = max(float(seed_minor), 1e-6)
        seed_gap = max(float(seed_major) - seed_minor, 1e-6)

        def residual_fn(params: np.ndarray) -> np.ndarray:
            center = seed_center + params[:3]
            axis_dir = _unit(_rotate_vector_by_rotvec(seed_axis, params[3:6]))
            minor_radius = float(np.exp(params[6]))
            major_radius = minor_radius + float(np.exp(params[7]))
            residual_signed = _torus_residual(fit_points, center, axis_dir, major_radius, minor_radius)
            return fit_weight_scale * residual_signed

        result = least_squares(
            residual_fn,
            np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(np.log(seed_minor)), float(np.log(seed_gap))],
                dtype=np.float64,
            ),
            loss="soft_l1",
            f_scale=f_scale,
            max_nfev=360,
        )
        if not result.success:
            residual_all = np.abs(_torus_residual(points, seed_center, seed_axis, seed_major, seed_minor))
            return {
                "center": seed_center.astype(np.float64),
                "axis_dir": seed_axis.astype(np.float64),
                "major_radius": float(seed_major),
                "minor_radius": float(seed_minor),
            }, residual_all.astype(np.float64)

        center = seed_center + result.x[:3]
        axis_dir = _unit(_rotate_vector_by_rotvec(seed_axis, result.x[3:6]))
        minor_radius = float(np.exp(result.x[6]))
        major_radius = minor_radius + float(np.exp(result.x[7]))
        residual_all = np.abs(_torus_residual(points, center, axis_dir, major_radius, minor_radius))
        return {
            "center": center.astype(np.float64),
            "axis_dir": axis_dir.astype(np.float64),
            "major_radius": major_radius,
            "minor_radius": minor_radius,
        }, residual_all.astype(np.float64)

    fitted_params, residual = _solve(points, weights, init_center, init_axis, init_major, init_minor)
    refine_mask = _robust_residual_mask(residual)
    if int(refine_mask.sum()) >= 12 and int(refine_mask.sum()) < len(points):
        fitted_params, residual = _solve(
            points[refine_mask],
            weights[refine_mask],
            np.asarray(fitted_params["center"], dtype=np.float64),
            _unit(np.asarray(fitted_params["axis_dir"], dtype=np.float64)),
            float(fitted_params["major_radius"]),
            float(fitted_params["minor_radius"]),
        )
    return fitted_params, residual


def _fit_surface(
    surface_type: str,
    points: np.ndarray,
    tri_normals: np.ndarray,
    weights: np.ndarray,
    init_params: dict,
) -> tuple[dict, np.ndarray]:
    """Dispatch surface fitting to the appropriate analytic solver."""
    if surface_type == "plane":
        return _fit_plane(points, tri_normals, weights)
    if surface_type == "cylinder":
        return _fit_cylinder(points, tri_normals, weights)
    if surface_type == "sphere":
        return _fit_sphere(points, weights, init_params)
    if surface_type == "cone":
        return _fit_cone(points, weights, init_params)
    if surface_type == "torus":
        return _fit_torus(points, weights, init_params)
    fitted_params = dict(init_params)
    residual = _reference_residual(surface_type, points, init_params)
    return fitted_params, residual


def _boundary_band_mm(tol_mm: float, support_gap_mm: float) -> float:
    """Return the width of the boundary band that needs stricter checks."""
    return max(float(tol_mm) * 2.5, float(support_gap_mm) * 0.6, 0.8)


def _boundary_normal_limit(surface_type: str) -> float:
    """Return the maximum allowed normal mismatch penalty near a boundary."""
    limit_deg = {
        "plane": 20.0,
        "cylinder": 28.0,
        "cone": 32.0,
        "sphere": 28.0,
        "torus": 30.0,
    }.get(surface_type, 30.0)
    return float(1.0 - np.cos(np.deg2rad(limit_deg)))


def _reference_residual(
    surface_type: str,
    points: np.ndarray,
    params: dict,
) -> np.ndarray:
    if surface_type == "plane":
        point = np.asarray(params["point"], dtype=np.float64)
        normal = _unit(np.asarray(params["normal"], dtype=np.float64))
        return np.abs((np.asarray(points, dtype=np.float64) - point[None, :]) @ normal)

    if surface_type == "cylinder":
        axis_origin = np.asarray(params["axis_origin"], dtype=np.float64)
        axis_dir = _unit(np.asarray(params["axis_dir"], dtype=np.float64))
        radius = float(params["radius"])
        radial = np.asarray(points, dtype=np.float64) - axis_origin[None, :]
        radial = radial - np.outer(radial @ axis_dir, axis_dir)
        return np.abs(np.linalg.norm(radial, axis=1) - radius)

    if surface_type == "sphere":
        center = np.asarray(params["center"], dtype=np.float64)
        radius = float(params["radius"])
        return np.abs(np.linalg.norm(points - center[None, :], axis=1) - radius)

    if surface_type == "cone":
        apex = np.asarray(params["apex"], dtype=np.float64)
        axis_dir = _unit(np.asarray(params["axis_dir"], dtype=np.float64))
        semi_angle = float(params["semi_angle_rad"])
        vec = points - apex[None, :]
        axial = vec @ axis_dir
        radial = np.linalg.norm(vec - np.outer(axial, axis_dir), axis=1)
        return np.abs(radial - np.abs(axial) * np.tan(semi_angle))

    if surface_type == "torus":
        center = np.asarray(params["center"], dtype=np.float64)
        axis_dir = _unit(np.asarray(params["axis_dir"], dtype=np.float64))
        major_radius = float(params["major_radius"])
        minor_radius = float(params["minor_radius"])
        vec = points - center[None, :]
        axial = vec @ axis_dir
        radial = np.linalg.norm(vec - np.outer(axial, axis_dir), axis=1)
        tube = np.sqrt((radial - major_radius) ** 2 + axial ** 2)
        return np.abs(tube - minor_radius)

    raise ValueError(f"Unsupported reference residual surface: {surface_type}")


def _surface_normal_penalty(
    surface_type: str,
    points: np.ndarray,
    tri_normals: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Measure how incompatible STL triangle normals are with a fitted surface."""
    points = np.asarray(points, dtype=np.float64)
    tri_normals = np.asarray(tri_normals, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)

    if surface_type == "plane":
        normal = _unit(np.asarray(params["normal"], dtype=np.float64))
        return (1.0 - np.abs(tri_normals @ normal)).astype(np.float64)

    if surface_type == "cylinder":
        axis_origin = np.asarray(params["axis_origin"], dtype=np.float64)
        axis_dir = _unit(np.asarray(params["axis_dir"], dtype=np.float64))
        radial = points - axis_origin[None, :]
        radial = radial - np.outer(radial @ axis_dir, axis_dir)
        radial_dir = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        return (1.0 - np.abs(np.sum(tri_normals * radial_dir, axis=1))).astype(np.float64)

    if surface_type == "sphere":
        center = np.asarray(params["center"], dtype=np.float64)
        radial_dir = points - center[None, :]
        radial_dir /= np.maximum(np.linalg.norm(radial_dir, axis=1, keepdims=True), 1e-12)
        return (1.0 - np.abs(np.sum(tri_normals * radial_dir, axis=1))).astype(np.float64)

    if surface_type == "cone":
        apex = np.asarray(params["apex"], dtype=np.float64)
        axis_dir = _unit(np.asarray(params["axis_dir"], dtype=np.float64))
        tan_angle = max(float(np.tan(float(params["semi_angle_rad"]))), 1e-6)
        vec = points - apex[None, :]
        axial = vec @ axis_dir
        radial = vec - np.outer(axial, axis_dir)
        radial_dir = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        normal = radial_dir - np.sign(axial)[:, None] * tan_angle * axis_dir[None, :]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        return (1.0 - np.abs(np.sum(tri_normals * normal, axis=1))).astype(np.float64)

    if surface_type == "torus":
        center = np.asarray(params["center"], dtype=np.float64)
        axis_dir = _unit(np.asarray(params["axis_dir"], dtype=np.float64))
        major_radius = float(params["major_radius"])
        vec = points - center[None, :]
        axial = vec @ axis_dir
        radial = vec - np.outer(axial, axis_dir)
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_dir = radial / np.maximum(radial_norm, 1e-12)
        normal = (radial_norm - major_radius) * radial_dir + axial[:, None] * axis_dir[None, :]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        return (1.0 - np.abs(np.sum(tri_normals * normal, axis=1))).astype(np.float64)

    return np.zeros(len(points), dtype=np.float64)


def _mesh_boundary_samples(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Sample vertices and edge midpoints on the boundary of a trimmed face mesh."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(triangles) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return np.zeros((0, 3), dtype=np.float64)

    boundary_vertices = sorted({idx for edge in boundary_edges for idx in edge})
    samples = [vertices[np.asarray(boundary_vertices, dtype=np.int32)]]
    edge_midpoints = np.asarray(
        [(vertices[u] + vertices[v]) * 0.5 for u, v in boundary_edges],
        dtype=np.float64,
    )
    if len(edge_midpoints):
        samples.append(edge_midpoints)
    return np.vstack(samples).astype(np.float64)


def _distance_to_face_scene(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
    closest = scene.compute_closest_points(query)["points"].numpy().astype(np.float64)
    return np.linalg.norm(np.asarray(points, dtype=np.float64) - closest, axis=1).astype(np.float64)


def _select_best_support_component(
    support_triangles: np.ndarray,
    cache: TriangleCache,
    face_scene: o3d.t.geometry.RaycastingScene,
    min_keep: int,
) -> np.ndarray:
    """Keep the strongest connected component inside a local support set."""
    support_triangles = np.asarray(support_triangles, dtype=np.int32)
    if support_triangles.size == 0:
        return support_triangles

    components = _connected_components_from_subset(support_triangles, cache.tri_neighbors)
    if len(components) <= 1:
        return support_triangles

    best_comp = support_triangles
    best_score = -float("inf")
    for comp in components:
        comp_weights = np.maximum(cache.tri_areas[comp], 1e-6)
        comp_centers = cache.tri_centers[comp]
        query = o3d.core.Tensor(comp_centers.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        closest = face_scene.compute_closest_points(query)["points"].numpy().astype(np.float64)
        dist_mean = float(np.average(np.linalg.norm(comp_centers - closest, axis=1), weights=comp_weights))
        area = float(comp_weights.sum())
        score = area - 0.35 * dist_mean * np.sqrt(max(area, 1.0))
        if score > best_score:
            best_score = score
            best_comp = comp

    if best_comp.size < max(min_keep, support_triangles.size // 4):
        return support_triangles
    return np.asarray(best_comp, dtype=np.int32)


def _purify_reference_support(
    face: TransformedStepFace,
    support_triangles: np.ndarray,
    cache: TriangleCache,
    face_scene: o3d.t.geometry.RaycastingScene,
    residual_limit: float,
    min_keep: int,
) -> np.ndarray:
    """Drop triangles that are clearly incompatible with the local STEP face."""
    support_triangles = np.asarray(support_triangles, dtype=np.int32)
    if support_triangles.size == 0:
        return support_triangles

    residual = _reference_residual(
        face.face.surface_type,
        cache.tri_centers[support_triangles],
        face.params,
    )
    filtered = support_triangles[residual <= float(residual_limit)].astype(np.int32)
    if filtered.size < max(min_keep, 8):
        return support_triangles
    return _select_best_support_component(filtered, cache, face_scene, min_keep=min_keep)


@dataclass
class StepSTLFitSession:
    """End-to-end STEP-guided fitting session for one STEP/STL pair."""
    step_path: str
    scan_stl_path: str
    linear_deflection: float = 0.5
    registration_config: RegistrationConfig = field(default_factory=RegistrationConfig)

    def __post_init__(self) -> None:
        self.step_faces: list[StepAnalyticFace] = []
        self.step_mesh: o3d.geometry.TriangleMesh | None = None
        self.scan_mesh: o3d.geometry.TriangleMesh | None = None
        self.scan_cache: TriangleCache | None = None
        self.T_step_to_scan = np.eye(4, dtype=np.float64)
        self.registration_info: dict = {}
        self.transformed_faces: list[TransformedStepFace] = []
        self._candidate_cache: dict[tuple[int, float], np.ndarray] = {}
        self._face_scene_cache: dict[int, o3d.t.geometry.RaycastingScene] = {}
        self._face_boundary_tree_cache: dict[int, cKDTree | None] = {}

    def load(self) -> None:
        """Load STEP faces, STL mesh, and the coarse STEP-to-STL transform."""
        self.step_faces, self.step_mesh = extract_step_analytic_faces(
            self.step_path,
            linear_deflection=self.linear_deflection,
        )
        if not self.step_faces:
            raise RuntimeError("No analytic STEP faces extracted.")

        self.scan_mesh = load_stl_mesh(self.scan_stl_path)
        self.scan_cache = _build_triangle_cache(self.scan_mesh)
        self.T_step_to_scan, self.registration_info = registration_coarse(
            scan_mesh=self.scan_mesh,
            cad_mesh=self.step_mesh,
            config=self.registration_config.as_namespace(),
        )
        self.transformed_faces = [_transform_face(face, self.T_step_to_scan) for face in self.step_faces]
        self._candidate_cache.clear()
        self._face_scene_cache.clear()
        self._face_boundary_tree_cache.clear()
        log(
            f"STEP-to-STL session loaded: faces={len(self.transformed_faces)} "
            f"scan_tris={len(self.scan_cache.triangles)}"
        )

    def ensure_loaded(self) -> None:
        """Lazily initialize the session before any fitting or picking."""
        if self.scan_cache is None or self.step_mesh is None or not self.transformed_faces:
            self.load()

    def face_labels(self) -> list[str]:
        self.ensure_loaded()
        return [
            f"Face {face.face.id} [{face.face.surface_type}] area={face.face.area_mm2:.1f}"
            for face in self.transformed_faces
        ]

    def _face_scene(self, face_id: int) -> o3d.t.geometry.RaycastingScene:
        if face_id not in self._face_scene_cache:
            face = self.transformed_faces[face_id]
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(face.mesh))
            self._face_scene_cache[face_id] = scene
        return self._face_scene_cache[face_id]

    def _face_boundary_tree(self, face_id: int) -> cKDTree | None:
        """Return a KD-tree built on the current face boundary samples."""
        if face_id not in self._face_boundary_tree_cache:
            samples = _mesh_boundary_samples(self.transformed_faces[face_id].mesh)
            self._face_boundary_tree_cache[face_id] = cKDTree(samples) if len(samples) else None
        return self._face_boundary_tree_cache[face_id]

    def _distance_to_face_boundary(self, face_id: int, points: np.ndarray) -> np.ndarray:
        """Query the distance from STL support points to the trimmed STEP face boundary."""
        tree = self._face_boundary_tree(face_id)
        if tree is None or len(points) == 0:
            return np.full(len(points), np.inf, dtype=np.float64)
        dist, _ = tree.query(np.asarray(points, dtype=np.float64), k=1)
        return np.asarray(dist, dtype=np.float64)

    def _candidate_triangles(self, face_id: int, support_gap_mm: float) -> np.ndarray:
        """Return STL triangles near a transformed STEP face."""
        key = (int(face_id), round(float(support_gap_mm), 3))
        if key in self._candidate_cache:
            return self._candidate_cache[key].copy()

        cache = self.scan_cache
        assert cache is not None
        face = self.transformed_faces[face_id]
        bmin = face.bbox_min - float(support_gap_mm)
        bmax = face.bbox_max + float(support_gap_mm)
        pts = cache.tri_centers
        aabb_mask = (
            (pts[:, 0] >= bmin[0]) & (pts[:, 0] <= bmax[0]) &
            (pts[:, 1] >= bmin[1]) & (pts[:, 1] <= bmax[1]) &
            (pts[:, 2] >= bmin[2]) & (pts[:, 2] <= bmax[2])
        )
        candidate_idx = np.flatnonzero(aabb_mask).astype(np.int32)
        if candidate_idx.size == 0:
            self._candidate_cache[key] = candidate_idx
            return candidate_idx

        scene = self._face_scene(face_id)
        query = o3d.core.Tensor(cache.tri_centers[candidate_idx].astype(np.float32), dtype=o3d.core.Dtype.Float32)
        closest = scene.compute_closest_points(query)["points"].numpy().astype(np.float64)
        dist = np.linalg.norm(cache.tri_centers[candidate_idx] - closest, axis=1)
        candidate_idx = candidate_idx[dist <= float(support_gap_mm)].astype(np.int32)
        self._candidate_cache[key] = candidate_idx
        return candidate_idx.copy()

    def pick_face_by_point(self, point_world: np.ndarray, max_distance_mm: float = 4.0) -> int | None:
        """Pick the nearest transformed STEP face to a 3D world-space point."""
        self.ensure_loaded()
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        pad = float(max_distance_mm)
        query = o3d.core.Tensor(point.reshape(1, 3).astype(np.float32), dtype=o3d.core.Dtype.Float32)

        best_index: int | None = None
        best_distance = float("inf")

        for face_index, face in enumerate(self.transformed_faces):
            if np.any(point < (face.bbox_min - pad)) or np.any(point > (face.bbox_max + pad)):
                continue

            scene = self._face_scene(face_index)
            closest = scene.compute_closest_points(query)["points"].numpy()[0].astype(np.float64)
            distance = float(np.linalg.norm(point - closest))
            if distance < best_distance:
                best_distance = distance
                best_index = face_index

        if best_index is None or best_distance > pad:
            return None
        return int(best_index)

    def pick_face_by_triangle(
        self,
        triangle_id: int,
        point_world: np.ndarray,
        thresholds: FitThresholds,
        max_distance_mm: float = 4.0,
    ) -> int | None:
        """Pick the best-matching STEP face using a hit STL triangle and hit point."""
        self.ensure_loaded()
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        triangle_id = int(triangle_id)
        pad = max(float(max_distance_mm), float(thresholds.support_gap_mm))
        query = o3d.core.Tensor(point.reshape(1, 3).astype(np.float32), dtype=o3d.core.Dtype.Float32)

        best_index: int | None = None
        best_score = float("inf")

        for face_index, face in enumerate(self.transformed_faces):
            if np.any(point < (face.bbox_min - pad)) or np.any(point > (face.bbox_max + pad)):
                continue

            candidates = self._candidate_triangles(face_index, thresholds.support_gap_mm)
            if candidates.size == 0 or not np.any(candidates == triangle_id):
                continue

            scene = self._face_scene(face_index)
            closest = scene.compute_closest_points(query)["points"].numpy()[0].astype(np.float64)
            distance = float(np.linalg.norm(point - closest))
            if distance > pad:
                continue

            residual = float(
                _reference_residual(
                    face.face.surface_type,
                    self.scan_cache.tri_centers[[triangle_id]],
                    face.params,
                )[0]
            )
            tol = max(float(thresholds.tolerance_for(face.face.surface_type)), 1e-6)
            score = distance / pad + 0.35 * residual / tol
            if score < best_score:
                best_score = score
                best_index = face_index

        if best_index is not None:
            return int(best_index)
        return self.pick_face_by_point(point_world=point_world, max_distance_mm=max_distance_mm)

    def _prepare_support_triangles(self, face_id: int, thresholds: FitThresholds) -> np.ndarray:
        """Build a cleaned local support region for one transformed STEP face."""
        cache = self.scan_cache
        assert cache is not None
        face = self.transformed_faces[face_id]
        face_scene = self._face_scene(face_id)
        support_triangles = self._candidate_triangles(face_id, thresholds.support_gap_mm)
        if face.face.surface_type == "plane":
            support_triangles = _purify_plane_support(
                face,
                support_triangles,
                cache,
                face_scene,
                min_keep=int(thresholds.min_support_triangles),
            )
        else:
            residual_limit = max(
                float(thresholds.tolerance_for(face.face.surface_type)) * 2.5,
                float(thresholds.support_gap_mm),
            )
            support_triangles = _purify_reference_support(
                face,
                support_triangles,
                cache,
                face_scene,
                residual_limit=residual_limit,
                min_keep=int(thresholds.min_support_triangles),
            )
        return np.asarray(support_triangles, dtype=np.int32)

    def _fit_face_with_support(
        self,
        face_id: int,
        support_triangles: np.ndarray,
        thresholds: FitThresholds,
    ) -> FaceFitResult:
        """Fit one STEP face from a prepared STL local support region."""
        cache = self.scan_cache
        assert cache is not None
        face = self.transformed_faces[face_id]
        face_scene = self._face_scene(face_id)

        support_triangles = np.asarray(support_triangles, dtype=np.int32)
        if support_triangles.size < thresholds.min_support_triangles:
            return _empty_result(face, "support triangles below minimum")

        points = cache.tri_centers[support_triangles]
        normals = cache.tri_normals[support_triangles]
        weights = np.maximum(cache.tri_areas[support_triangles], 1e-6)
        support_distances = _distance_to_face_scene(face_scene, points)
        boundary_distances = self._distance_to_face_boundary(face_id, points)

        tol = thresholds.tolerance_for(face.face.surface_type)
        fitted_params, residual = _fit_surface(
            face.face.surface_type,
            points,
            normals,
            weights,
            face.params,
        )
        normal_penalty = _surface_normal_penalty(
            face.face.surface_type,
            points,
            normals,
            fitted_params,
        )
        boundary_band = _boundary_band_mm(tol, thresholds.support_gap_mm)
        boundary_mask = boundary_distances <= boundary_band
        inlier_mask = residual <= tol
        inlier_mask[boundary_mask] &= normal_penalty[boundary_mask] <= _boundary_normal_limit(face.face.surface_type)
        if int(inlier_mask.sum()) >= max(int(thresholds.min_support_triangles), 8):
            refit_params, _ = _fit_surface(
                face.face.surface_type,
                points[inlier_mask],
                normals[inlier_mask],
                weights[inlier_mask],
                fitted_params,
            )
            refit_residual = _reference_residual(face.face.surface_type, points, refit_params)
            refit_normal_penalty = _surface_normal_penalty(
                face.face.surface_type,
                points,
                normals,
                refit_params,
            )
            refit_inlier_mask = refit_residual <= tol
            refit_inlier_mask[boundary_mask] &= (
                refit_normal_penalty[boundary_mask] <= _boundary_normal_limit(face.face.surface_type)
            )
            if int(refit_inlier_mask.sum()) >= max(int(thresholds.min_support_triangles), 8):
                fitted_params = refit_params
                residual = refit_residual
                normal_penalty = refit_normal_penalty
                inlier_mask = refit_inlier_mask

        inlier_triangles = support_triangles[inlier_mask].astype(np.int32)
        outlier_triangles = support_triangles[~inlier_mask].astype(np.int32)

        support_mesh = _submesh_from_triangles(cache.mesh, support_triangles)
        inlier_mesh = _submesh_from_triangles(cache.mesh, inlier_triangles)
        outlier_mesh = _submesh_from_triangles(cache.mesh, outlier_triangles)

        support_area = float(cache.tri_areas[support_triangles].sum())
        inlier_area = float(cache.tri_areas[inlier_triangles].sum()) if inlier_triangles.size else 0.0
        outlier_area = float(cache.tri_areas[outlier_triangles].sum()) if outlier_triangles.size else 0.0

        return FaceFitResult(
            face_id=face.face.id,
            surface_type=face.face.surface_type,
            status="ok",
            message="fit complete",
            transformed_face_mesh=o3d.geometry.TriangleMesh(face.mesh),
            support_mesh=support_mesh,
            inlier_mesh=inlier_mesh,
            outlier_mesh=outlier_mesh,
            support_triangles=support_triangles,
            support_residuals=np.asarray(residual, dtype=np.float64),
            support_distances=np.asarray(support_distances, dtype=np.float64),
            inlier_triangles=inlier_triangles,
            outlier_triangles=outlier_triangles,
            fitted_params=fitted_params,
            support_area_mm2=support_area,
            inlier_area_mm2=inlier_area,
            outlier_area_mm2=outlier_area,
            residual_mean_mm=float(np.mean(residual)) if residual.size else 0.0,
            residual_p95_mm=float(np.percentile(residual, 95)) if residual.size else 0.0,
            residual_max_mm=float(np.max(residual)) if residual.size else 0.0,
            inlier_ratio=float(np.mean(inlier_mask)) if residual.size else 0.0,
        )

    def _resolve_global_triangle_ownership(
        self,
        results: list[FaceFitResult],
        thresholds: FitThresholds,
    ) -> dict[int, np.ndarray]:
        """Assign every contested STL triangle to a single best-fitting face."""
        best_owner: dict[int, tuple[float, int]] = {}
        cache = self.scan_cache
        assert cache is not None

        for face_index, result in enumerate(results):
            if result.status != "ok" or result.support_triangles.size == 0:
                continue
            tol = max(thresholds.tolerance_for(result.surface_type), 1e-6)
            gap = max(float(thresholds.support_gap_mm), 1e-6)
            support_triangles = result.support_triangles.astype(np.int32)
            support_points = cache.tri_centers[support_triangles]
            support_normals = cache.tri_normals[support_triangles]
            normal_penalty = _surface_normal_penalty(
                result.surface_type,
                support_points,
                support_normals,
                result.fitted_params,
            )
            boundary_distance = self._distance_to_face_boundary(face_index, support_points)
            boundary_band = _boundary_band_mm(tol, thresholds.support_gap_mm)
            for tri_id, residual, dist, normal_cost, boundary_dist in zip(
                support_triangles.tolist(),
                result.support_residuals.tolist(),
                result.support_distances.tolist(),
                normal_penalty.tolist(),
                boundary_distance.tolist(),
            ):
                boundary_scale = 0.45
                if float(boundary_dist) <= boundary_band:
                    boundary_scale = 0.90
                score = float(residual) / tol + 0.20 * float(dist) / gap + boundary_scale * float(normal_cost)
                prev = best_owner.get(int(tri_id))
                if prev is None or score < prev[0]:
                    best_owner[int(tri_id)] = (score, face_index)

        assigned: dict[int, list[int]] = {face_index: [] for face_index in range(len(results))}
        for tri_id, (_, face_index) in best_owner.items():
            assigned[face_index].append(int(tri_id))
        return {
            face_index: np.asarray(sorted(tri_ids), dtype=np.int32)
            for face_index, tri_ids in assigned.items()
        }

    def analyze_face(self, face_id: int, thresholds: FitThresholds) -> FaceFitResult:
        """Run local cleaning and fitting for a single STEP face."""
        self.ensure_loaded()
        support_triangles = self._prepare_support_triangles(face_id, thresholds)
        return self._fit_face_with_support(face_id, support_triangles, thresholds)

    def analyze_all_faces(self, thresholds: FitThresholds) -> list[FaceFitResult]:
        """Run all faces, then resolve triangle ownership globally and refit."""
        self.ensure_loaded()
        local_supports = [
            self._prepare_support_triangles(face_index, thresholds)
            for face_index in range(len(self.transformed_faces))
        ]
        initial_results = [
            self._fit_face_with_support(face_index, support_triangles, thresholds)
            for face_index, support_triangles in enumerate(local_supports)
        ]

        ownership = self._resolve_global_triangle_ownership(initial_results, thresholds)
        final_results: list[FaceFitResult] = []
        for face_index, local_support in enumerate(local_supports):
            owned = ownership.get(face_index, np.zeros(0, dtype=np.int32))
            final_support = owned
            if final_support.size < thresholds.min_support_triangles:
                final_support = local_support
            final_results.append(self._fit_face_with_support(face_index, final_support, thresholds))
        return final_results
