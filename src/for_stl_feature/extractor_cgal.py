import numpy as np
import open3d as o3d
from typing import Tuple, List

from .cpp_moudle import cgal_ransac
from .core_types import ScanPlaneFeature, ScanCylinderFeature, BaseFeatureExtractor
from .scan_features_stl import (
    load_stl_mesh,
    _unit,
    _build_triangle_adjacency,
    _submesh_from_triangles,
    _triangle_areas_and_centroids,
)


def _sample_surface_points(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_areas: np.ndarray,
    tri_normals: np.ndarray,
    resolution_mm: float,
    sample_multiplier: float,
    min_points: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_area = float(tri_areas.sum())
    if total_area <= 1e-12 or len(triangles) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    area_per_sample = max(float(resolution_mm) ** 2, 1e-6)
    target_samples = int(total_area / area_per_sample * max(sample_multiplier, 1e-3))
    target_samples = int(np.clip(target_samples, min_points, max_points))

    rng = np.random.default_rng(seed)
    tri_prob = tri_areas / total_area
    tri_ids = rng.choice(len(triangles), size=target_samples, p=tri_prob)

    r1 = np.sqrt(rng.random(target_samples))
    r2 = rng.random(target_samples)

    p0 = vertices[triangles[tri_ids, 0]]
    p1 = vertices[triangles[tri_ids, 1]]
    p2 = vertices[triangles[tri_ids, 2]]

    points = (
        (1.0 - r1)[:, None] * p0
        + (r1 * (1.0 - r2))[:, None] * p1
        + (r1 * r2)[:, None] * p2
    )
    normals = tri_normals[tri_ids]
    return points.astype(np.float64), normals.astype(np.float64)


def _sorted_shape_proposals(index_lists, param_lists):
    proposals = []
    for idx_list, param in zip(index_lists, param_lists):
        idx = np.asarray(idx_list, dtype=np.int32)
        if idx.size == 0:
            continue
        proposals.append((int(idx.size), idx, param))
    proposals.sort(key=lambda item: item[0], reverse=True)
    return proposals


def _split_triangle_components(mask: np.ndarray, adjacency: list[list[int]]) -> list[np.ndarray]:
    tri_idx = np.flatnonzero(mask)
    if tri_idx.size == 0:
        return []

    seen = np.zeros(mask.shape[0], dtype=bool)
    components: list[np.ndarray] = []

    for seed in tri_idx:
        if seen[seed]:
            continue

        stack = [int(seed)]
        seen[seed] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adjacency[cur]:
                if mask[nb] and not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)

        components.append(np.asarray(comp, dtype=np.int32))

    return components


def _plane_triangle_mask(
    tri_centers: np.ndarray,
    tri_normals: np.ndarray,
    normal: np.ndarray,
    d: float,
    dist_tol: float,
    cos_tol: float,
) -> np.ndarray:
    dist = np.abs(tri_centers @ normal + d)
    align = np.abs(tri_normals @ normal)
    return (dist <= dist_tol) & (align >= cos_tol)


def _cylinder_triangle_mask(
    tri_centers: np.ndarray,
    tri_normals: np.ndarray,
    axis_origin: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    dist_tol: float,
    cos_tol: float,
    sin_tol: float,
) -> np.ndarray:
    vec = tri_centers - axis_origin
    radial = vec - np.outer(vec @ axis_dir, axis_dir)
    radial_norm = np.linalg.norm(radial, axis=1)
    radial_dir = radial / np.maximum(radial_norm[:, None], 1e-12)

    radial_resid = np.abs(radial_norm - radius)
    normal_align = np.abs(np.sum(tri_normals * radial_dir, axis=1))
    axis_orth = np.abs(tri_normals @ axis_dir)

    return (radial_resid <= dist_tol) & (normal_align >= cos_tol) & (axis_orth <= sin_tol)


def _estimate_plane_from_component(
    tri_centers: np.ndarray,
    tri_normals: np.ndarray,
    tri_areas: np.ndarray,
    comp: np.ndarray,
    fallback_normal: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float, float]:
    weights = tri_areas[comp]
    centers = tri_centers[comp]
    normals = tri_normals[comp]

    accum_normal = (normals * weights[:, None]).sum(axis=0)
    if np.linalg.norm(accum_normal) <= 1e-12:
        normal = _unit(fallback_normal.astype(np.float64))
    else:
        normal = _unit(accum_normal.astype(np.float64))

    if float(np.dot(normal, fallback_normal)) < 0.0:
        normal = -normal

    centroid = np.average(centers, axis=0, weights=weights)
    d = -float(normal @ centroid)

    dist = centers @ normal + d
    rmse = float(np.sqrt(np.average(dist * dist, weights=weights)))
    p95 = float(np.percentile(np.abs(dist), 95))
    return normal, d, centroid, rmse, p95


def _estimate_cylinder_from_component(
    tri_centers: np.ndarray,
    tri_areas: np.ndarray,
    comp: np.ndarray,
    fallback_origin: np.ndarray,
    fallback_axis_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    axis_dir = _unit(fallback_axis_dir.astype(np.float64))
    centers = tri_centers[comp]
    weights = tri_areas[comp]

    offset_along_axis = np.average((centers - fallback_origin) @ axis_dir, weights=weights)
    axis_origin = fallback_origin + offset_along_axis * axis_dir

    vec = centers - axis_origin
    radial = vec - np.outer(vec @ axis_dir, axis_dir)
    radial_norm = np.linalg.norm(radial, axis=1)

    radius = float(np.average(radial_norm, weights=weights))
    resid = radial_norm - radius
    rmse = float(np.sqrt(np.average(resid * resid, weights=weights)))
    return axis_origin, axis_dir, radius, rmse


def _mean_component_normal(
    tri_normals: np.ndarray,
    tri_areas: np.ndarray,
    comp: np.ndarray,
) -> np.ndarray:
    weighted = (tri_normals[comp] * tri_areas[comp][:, None]).sum(axis=0)
    if np.linalg.norm(weighted) <= 1e-12:
        weighted = tri_normals[comp].mean(axis=0)
    return _unit(weighted.astype(np.float64))


def _expand_plane_component(
    adjacency: list[list[int]],
    tri_centers: np.ndarray,
    tri_normals: np.ndarray,
    base_component: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    dist_tol: float,
    cos_tol: float,
    assigned: np.ndarray,
) -> np.ndarray:
    in_region = np.zeros(assigned.shape[0], dtype=bool)
    in_region[base_component] = True
    queue = [int(x) for x in base_component.tolist()]

    while queue:
        cur = queue.pop()
        for nb in adjacency[cur]:
            if assigned[nb] or in_region[nb]:
                continue
            if abs(float(tri_normals[nb] @ plane_normal)) < cos_tol:
                continue
            if abs(float(tri_centers[nb] @ plane_normal + plane_d)) > dist_tol:
                continue
            in_region[nb] = True
            queue.append(int(nb))

    return np.flatnonzero(in_region).astype(np.int32)


def _grow_normal_region(
    adjacency: list[list[int]],
    tri_normals: np.ndarray,
    seed: int,
    available: np.ndarray,
    cos_tol: float,
) -> np.ndarray:
    visited = np.zeros(available.shape[0], dtype=bool)
    seed_n = tri_normals[seed]
    stack = [int(seed)]
    visited[seed] = True
    region = []

    while stack:
        cur = stack.pop()
        region.append(cur)
        cur_n = tri_normals[cur]
        for nb in adjacency[cur]:
            if visited[nb] or not available[nb]:
                continue
            nb_n = tri_normals[nb]
            if abs(float(cur_n @ nb_n)) >= cos_tol and abs(float(seed_n @ nb_n)) >= cos_tol:
                visited[nb] = True
                stack.append(int(nb))

    return np.asarray(region, dtype=np.int32)


class CGALExtractor(BaseFeatureExtractor):
    """
    Mesh-aware wrapper around the CGAL Efficient RANSAC backend.

    CGAL still performs the primitive proposal step, but the final triangle
    assignment is rebuilt on the cleaned mesh and then split by connected
    components. This avoids the previous "snowflake" result where many
    disconnected triangles sharing a similar analytic primitive were merged
    into a single feature.
    """

    def extract(
        self, scan_stl: str, **kwargs
    ) -> Tuple[
        List[ScanPlaneFeature],
        List[ScanCylinderFeature],
        np.ndarray,
        o3d.geometry.TriangleMesh,
    ]:
        resolution = float(kwargs.get("resolution_mm", 1.3))
        epsilon = float(kwargs.get("plane_dist_tol", 1.0))
        plane_angle_deg = float(kwargs.get("plane_angle_deg", 6.0))
        plane_min_area = float(kwargs.get("plane_min_area", 100.0))
        plane_min_triangles = int(kwargs.get("plane_min_triangles", 20))
        plane_p95_tol = float(kwargs.get("plane_p95_tol", epsilon))
        plane_expand_dist_tol = float(kwargs.get("plane_expand_dist_tol", max(epsilon * 1.25, epsilon + 0.2)))
        plane_expand_angle_deg = float(kwargs.get("plane_expand_angle_deg", plane_angle_deg + 2.0))
        plane_recover_angle_deg = float(kwargs.get("plane_recover_angle_deg", plane_angle_deg + 2.0))
        plane_recover_min_triangles = int(kwargs.get("plane_recover_min_triangles", max(plane_min_triangles, 30)))

        cyl_min_area = float(kwargs.get("cyl_min_area", 80.0))
        cyl_min_triangles = int(kwargs.get("cyl_min_triangles", 30))
        cyl_dist_tol = float(kwargs.get("cyl_dist_tol", max(epsilon * 1.5, 1.5)))
        cyl_angle_deg = float(kwargs.get("cyl_angle_deg", 12.0))
        cyl_rmse_tol = float(kwargs.get("cyl_rmse_tol", 2.0))
        cyl_to_plane_p95_tol = float(kwargs.get("cyl_to_plane_p95_tol", max(plane_p95_tol * 1.1, epsilon)))

        probability = float(kwargs.get("probability", 0.05))
        cluster_epsilon = float(kwargs.get("cluster_epsilon", max(resolution * 1.5, epsilon)))
        normal_threshold = float(kwargs.get("normal_threshold", 0.9))
        normal_threshold = float(np.clip(normal_threshold, 0.0, 1.0))

        sample_multiplier = float(kwargs.get("sample_multiplier", 1.0))
        sample_min_points = int(kwargs.get("sample_min_points", 20000))
        sample_max_points = int(kwargs.get("sample_max_points", 120000))
        sample_seed = int(kwargs.get("sample_seed", 0))
        random_seed = int(kwargs.get("random_seed", sample_seed))

        min_area = min(plane_min_area, cyl_min_area)
        min_points_default = max(
            50,
            int(round(min_area / max(resolution * resolution, 1e-6))),
        )
        min_points = int(kwargs.get("min_points", min_points_default))

        mesh = load_stl_mesh(scan_stl)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        tri_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
        tri_areas, tri_centers = _triangle_areas_and_centroids(vertices, triangles)
        adjacency = _build_triangle_adjacency(triangles)

        sample_points, sample_normals = _sample_surface_points(
            vertices=vertices,
            triangles=triangles,
            tri_areas=tri_areas,
            tri_normals=tri_normals,
            resolution_mm=resolution,
            sample_multiplier=sample_multiplier,
            min_points=sample_min_points,
            max_points=sample_max_points,
            seed=sample_seed,
        )
        if len(sample_points) == 0:
            return [], [], np.ones(len(triangles), dtype=bool), mesh

        print(
            "\n[CGAL Extractor] Sampling mesh surface and sending proposals to the C++ backend..."
        )
        c_planes_idx, c_planes_param, c_cyls_idx, c_cyls_param = cgal_ransac.extract_shapes(
            np.ascontiguousarray(sample_points, dtype=np.float64),
            np.ascontiguousarray(sample_normals, dtype=np.float64),
            probability,
            min_points,
            epsilon,
            cluster_epsilon,
            normal_threshold,
            random_seed,
        )

        plane_cos_tol = float(np.cos(np.deg2rad(plane_angle_deg)))
        plane_expand_cos_tol = float(np.cos(np.deg2rad(plane_expand_angle_deg)))
        plane_recover_cos_tol = float(np.cos(np.deg2rad(plane_recover_angle_deg)))
        cyl_cos_tol = float(np.cos(np.deg2rad(cyl_angle_deg)))
        cyl_sin_tol = float(np.sin(np.deg2rad(cyl_angle_deg)))

        assigned = np.zeros(len(triangles), dtype=bool)
        remaining_mask = np.ones(len(triangles), dtype=bool)
        planes: list[ScanPlaneFeature] = []
        cyls: list[ScanCylinderFeature] = []
        shape_id = 0

        plane_proposals = _sorted_shape_proposals(c_planes_idx, c_planes_param)
        for _, _, param in plane_proposals:
            raw_n = np.asarray(param[0:3], dtype=np.float64)
            n_norm = np.linalg.norm(raw_n)
            if n_norm <= 1e-12:
                continue

            normal = raw_n / n_norm
            d = float(param[3]) / n_norm

            candidate_mask = _plane_triangle_mask(
                tri_centers=tri_centers,
                tri_normals=tri_normals,
                normal=normal,
                d=d,
                dist_tol=epsilon,
                cos_tol=plane_cos_tol,
            ) & (~assigned)

            for comp in _split_triangle_components(candidate_mask, adjacency):
                if comp.size < plane_min_triangles:
                    continue

                area = float(tri_areas[comp].sum())
                if area < plane_min_area:
                    continue

                fit_n, fit_d, centroid, rmse, p95 = _estimate_plane_from_component(
                    tri_centers=tri_centers,
                    tri_normals=tri_normals,
                    tri_areas=tri_areas,
                    comp=comp,
                    fallback_normal=normal,
                )
                comp = _expand_plane_component(
                    adjacency=adjacency,
                    tri_centers=tri_centers,
                    tri_normals=tri_normals,
                    base_component=comp,
                    plane_normal=fit_n,
                    plane_d=fit_d,
                    dist_tol=plane_expand_dist_tol,
                    cos_tol=plane_expand_cos_tol,
                    assigned=assigned,
                )
                area = float(tri_areas[comp].sum())
                if area < plane_min_area or comp.size < plane_min_triangles:
                    continue
                fit_n, fit_d, centroid, rmse, p95 = _estimate_plane_from_component(
                    tri_centers=tri_centers,
                    tri_normals=tri_normals,
                    tri_areas=tri_areas,
                    comp=comp,
                    fallback_normal=fit_n,
                )
                if p95 > plane_p95_tol:
                    continue

                sub_mesh = _submesh_from_triangles(mesh, comp)
                planes.append(
                    ScanPlaneFeature(
                        id=shape_id,
                        tri_indices=comp.astype(np.int32),
                        mesh=sub_mesh,
                        normal=fit_n.astype(np.float64),
                        d=float(fit_d),
                        centroid=centroid.astype(np.float64),
                        area=area,
                        rmse=float(rmse),
                    )
                )
                assigned[comp] = True
                remaining_mask[comp] = False
                shape_id += 1

        recover_available = ~assigned
        seed_order = np.argsort(-tri_areas)
        for seed in seed_order:
            if not recover_available[seed]:
                continue

            region = _grow_normal_region(
                adjacency=adjacency,
                tri_normals=tri_normals,
                seed=int(seed),
                available=recover_available,
                cos_tol=plane_recover_cos_tol,
            )
            if region.size < plane_recover_min_triangles:
                recover_available[region] = False
                continue

            area = float(tri_areas[region].sum())
            if area < plane_min_area:
                recover_available[region] = False
                continue

            fallback_normal = _mean_component_normal(
                tri_normals=tri_normals,
                tri_areas=tri_areas,
                comp=region,
            )
            fit_n, fit_d, centroid, rmse, p95 = _estimate_plane_from_component(
                tri_centers=tri_centers,
                tri_normals=tri_normals,
                tri_areas=tri_areas,
                comp=region,
                fallback_normal=fallback_normal,
            )
            if p95 > plane_p95_tol:
                recover_available[region] = False
                continue

            region = _expand_plane_component(
                adjacency=adjacency,
                tri_centers=tri_centers,
                tri_normals=tri_normals,
                base_component=region,
                plane_normal=fit_n,
                plane_d=fit_d,
                dist_tol=plane_expand_dist_tol,
                cos_tol=plane_expand_cos_tol,
                assigned=assigned,
            )
            area = float(tri_areas[region].sum())
            if area < plane_min_area or region.size < plane_recover_min_triangles:
                recover_available[region] = False
                continue

            fit_n, fit_d, centroid, rmse, p95 = _estimate_plane_from_component(
                tri_centers=tri_centers,
                tri_normals=tri_normals,
                tri_areas=tri_areas,
                comp=region,
                fallback_normal=fit_n,
            )
            if p95 > plane_p95_tol:
                recover_available[region] = False
                continue

            sub_mesh = _submesh_from_triangles(mesh, region)
            planes.append(
                ScanPlaneFeature(
                    id=shape_id,
                    tri_indices=region.astype(np.int32),
                    mesh=sub_mesh,
                    normal=fit_n.astype(np.float64),
                    d=float(fit_d),
                    centroid=centroid.astype(np.float64),
                    area=area,
                    rmse=float(rmse),
                )
            )
            assigned[region] = True
            remaining_mask[region] = False
            recover_available[region] = False
            shape_id += 1

        cyl_proposals = _sorted_shape_proposals(c_cyls_idx, c_cyls_param)
        for _, _, param in cyl_proposals:
            axis_origin = np.asarray(param[0:3], dtype=np.float64)
            axis_dir = _unit(np.asarray(param[3:6], dtype=np.float64))
            radius = float(param[6])

            if not np.isfinite(radius) or radius <= 0.0:
                continue
            if np.linalg.norm(axis_dir) <= 1e-12:
                continue

            candidate_mask = _cylinder_triangle_mask(
                tri_centers=tri_centers,
                tri_normals=tri_normals,
                axis_origin=axis_origin,
                axis_dir=axis_dir,
                radius=radius,
                dist_tol=cyl_dist_tol,
                cos_tol=cyl_cos_tol,
                sin_tol=cyl_sin_tol,
            ) & (~assigned)

            for comp in _split_triangle_components(candidate_mask, adjacency):
                if comp.size < cyl_min_triangles:
                    continue

                area = float(tri_areas[comp].sum())
                if area < cyl_min_area:
                    continue

                fit_o, fit_v, fit_r, rmse = _estimate_cylinder_from_component(
                    tri_centers=tri_centers,
                    tri_areas=tri_areas,
                    comp=comp,
                    fallback_origin=axis_origin,
                    fallback_axis_dir=axis_dir,
                )
                if not np.isfinite(rmse) or not np.isfinite(fit_r):
                    continue
                if fit_r < 0.5 or fit_r > 5000.0:
                    continue
                if rmse > cyl_rmse_tol:
                    continue

                plane_like_normal = _mean_component_normal(
                    tri_normals=tri_normals,
                    tri_areas=tri_areas,
                    comp=comp,
                )
                plane_n, plane_d, plane_centroid, plane_rmse, plane_p95 = _estimate_plane_from_component(
                    tri_centers=tri_centers,
                    tri_normals=tri_normals,
                    tri_areas=tri_areas,
                    comp=comp,
                    fallback_normal=plane_like_normal,
                )
                if plane_p95 <= cyl_to_plane_p95_tol and plane_rmse <= rmse * 1.25:
                    sub_mesh = _submesh_from_triangles(mesh, comp)
                    planes.append(
                        ScanPlaneFeature(
                            id=shape_id,
                            tri_indices=comp.astype(np.int32),
                            mesh=sub_mesh,
                            normal=plane_n.astype(np.float64),
                            d=float(plane_d),
                            centroid=plane_centroid.astype(np.float64),
                            area=area,
                            rmse=float(plane_rmse),
                        )
                    )
                    assigned[comp] = True
                    remaining_mask[comp] = False
                    shape_id += 1
                    continue

                if float(np.dot(fit_v, axis_dir)) < 0.0:
                    fit_v = -fit_v

                sub_mesh = _submesh_from_triangles(mesh, comp)
                cyls.append(
                    ScanCylinderFeature(
                        id=shape_id,
                        tri_indices=comp.astype(np.int32),
                        mesh=sub_mesh,
                        axis_origin=fit_o.astype(np.float64),
                        axis_dir=_unit(fit_v.astype(np.float64)),
                        radius=float(fit_r),
                        rmse=float(rmse),
                    )
                )
                assigned[comp] = True
                remaining_mask[comp] = False
                shape_id += 1

        return planes, cyls, remaining_mask, mesh
