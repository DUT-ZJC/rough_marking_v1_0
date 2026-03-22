from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from .core_types import ScanPlaneFeature, ScanCylinderFeature, BaseFeatureExtractor
from ..logging_utils import log


def load_stl_mesh(path: str) -> o3d.geometry.TriangleMesh:
    log(f"Loading STL mesh: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load STL: {path}")
    mesh.compute_vertex_normals()

    #处理平面
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _triangle_areas_and_centroids(V: np.ndarray, F: np.ndarray):
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(n, axis=1)
    c = (p0 + p1 + p2) / 3.0
    return area, c


def _build_triangle_adjacency(F: np.ndarray):
    """
    adjacency list by shared edges.
    returns: list[list[int]] of length n_tri
    """
    n_tri = F.shape[0]
    edge_map = {}
    for ti in range(n_tri):
        a, b, c = F[ti]
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            if u > v:
                u, v = v, u
            edge_map.setdefault((u, v), []).append(ti)

    adj = [[] for _ in range(n_tri)]
    for tris in edge_map.values():
        if len(tris) < 2:
            continue
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                t1, t2 = tris[i], tris[j]
                adj[t1].append(t2)
                adj[t2].append(t1)
    return adj


def _submesh_from_triangles(mesh: o3d.geometry.TriangleMesh, tri_idx: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Robust triangle-submesh extraction compatible with Open3D builds where select_by_index is vertex-based.
    tri_idx are triangle indices (faces).
    """
    tri_idx = np.asarray(tri_idx, dtype=np.int64)
    if tri_idx.size == 0:
        return o3d.geometry.TriangleMesh()

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles, dtype=np.int64)

    # guard
    tri_idx = tri_idx[(tri_idx >= 0) & (tri_idx < len(F))]
    if tri_idx.size == 0:
        return o3d.geometry.TriangleMesh()

    Fsub = F[tri_idx]  # (K,3)

    # build compact vertex set
    used = np.unique(Fsub.reshape(-1))
    used = used[(used >= 0) & (used < len(V))]
    if used.size == 0:
        return o3d.geometry.TriangleMesh()

    remap = {int(old): i for i, old in enumerate(used)}
    Fnew = np.vectorize(remap.get)(Fsub)

    Vnew = V[used]

    sub = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(Vnew.astype(np.float64)),
        o3d.utility.Vector3iVector(Fnew.astype(np.int32))
    )
    if len(sub.triangles) > 0:
        sub.compute_vertex_normals()
    return sub


def _fit_plane_from_points(P: np.ndarray):
    c = P.mean(axis=0)
    X = P - c
    C = X.T @ X
    w, v = np.linalg.eigh(C)
    n = v[:, 0]
    n = _unit(n)
    d = -float(n @ c)
    dist = (P @ n + d)
    rmse = float(np.sqrt(np.mean(dist ** 2)))
    return n, d, c, rmse


def _region_grow_planes(mesh: o3d.geometry.TriangleMesh,
                        angle_deg: float = 6.0,
                        dist_tol: float = 1.0,
                        min_area: float = 100.0):
    """
    Mesh plane extraction via region growing.
    angle_deg: max normal angle between neighboring triangles
    dist_tol: max vertex-to-plane distance when expanding region (mm)
    min_area: keep plane region if area >= min_area (mm^2)
    """
    mesh = o3d.geometry.TriangleMesh(mesh)  # copy
    mesh.compute_triangle_normals()

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    N = np.asarray(mesh.triangle_normals)

    tri_area, tri_cent = _triangle_areas_and_centroids(V, F)
    adj = _build_triangle_adjacency(F)

    cos_th = np.cos(np.deg2rad(angle_deg))
    visited = np.zeros(len(F), dtype=bool)

    planes = []
    pid = 0

    for seed in range(len(F)):
        if visited[seed]:
            continue
        # start region
        stack = [seed]
        region = []

        visited[seed] = True
        seed_n = _unit(N[seed])

        # grow by normal similarity first
        while stack:
            t = stack.pop()
            region.append(t)
            nt = _unit(N[t])
            for nb in adj[t]:
                if visited[nb]:
                    continue
                nnb = _unit(N[nb])
                if abs(float(nt @ nnb)) >= cos_th and abs(float(seed_n @ nnb)) >= cos_th:
                    visited[nb] = True
                    stack.append(nb)

        region = np.array(region, dtype=np.int32)
        if region.size < 50:
            continue

        # fit plane on region vertices
        verts = np.unique(F[region].reshape(-1))
        P = V[verts]
        n, d, c, rmse = _fit_plane_from_points(P)

        # refine: reject triangles whose vertices too far from plane
        dist_v = np.abs(P @ n + d)
        if float(np.percentile(dist_v, 95)) > dist_tol:
            # too warped for plane -> skip
            continue

        area = float(tri_area[region].sum())
        if area < min_area:
            continue

        planes.append((pid, region, n, d, c, area, rmse))
        pid += 1

    return planes


def _axis_from_normals(normals: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    cylinder axis direction v satisfies n_i^T v ~ 0, so v is eigenvector of sum w n n^T with smallest eigenvalue.
    """
    if weights is None:
        weights = np.ones(len(normals), dtype=np.float64)
    M = np.zeros((3, 3), dtype=np.float64)
    for n, w in zip(normals, weights):
        n = _unit(n)
        M += w * np.outer(n, n)
    wv, vv = np.linalg.eigh(M)
    v = vv[:, 0]
    return _unit(v)


def _fit_circle_2d(xy: np.ndarray):
    """
    Simple algebraic circle fit (Kasa). Good enough for MVP.
    returns center (2,), radius, rmse
    """
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.stack([2*x, 2*y, np.ones_like(x)], axis=1)
    b = x*x + y*y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(1e-12, c + cx*cx + cy*cy))
    d = np.sqrt((x - cx)**2 + (y - cy)**2) - r
    rmse = float(np.sqrt(np.mean(d**2)))
    return np.array([cx, cy], dtype=np.float64), float(r), rmse


def _fit_cylinder_from_region(mesh: o3d.geometry.TriangleMesh, tri_idx: np.ndarray):
    """
    Fit cylinder parameters (axis, radius) from a triangle region.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    mesh.compute_triangle_normals()
    Ntri = np.asarray(mesh.triangle_normals)

    tri_idx = tri_idx.astype(np.int32)
    tri_verts = np.unique(F[tri_idx].reshape(-1))
    P = V[tri_verts]

    # axis from triangle normals
    normals = Ntri[tri_idx]
    v = _axis_from_normals(normals)

    # build orthonormal basis (u,w,v)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(tmp @ v)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(v, tmp))
    w = _unit(np.cross(v, u))

    # project points into plane perpendicular to v => 2D circle fit
    # remove v component: q = p - v(v^T p)
    # then 2D coords: [q·u, q·w]
    q = P - np.outer(P @ v, v)
    xy = np.stack([q @ u, q @ w], axis=1)
    c2, r, rmse2 = _fit_circle_2d(xy)

    # lift center back to 3D plane: c3 = u*c2x + w*c2y
    c3 = u * c2[0] + w * c2[1]

    # choose axis origin o: align to mean along v
    alpha = float(np.mean(P @ v))
    o = c3 + v * alpha

    # 3D residual: distance to axis - r
    dist = np.linalg.norm(np.cross(P - o, v), axis=1)
    resid = dist - r
    rmse3 = float(np.sqrt(np.mean(resid**2)))

    return o, v, r, rmse3


def extract_scan_planes_and_cylinders_from_mesh(
    scan_stl: str,
    plane_angle_deg: float = 6.0,
    plane_dist_tol: float = 1.0,
    plane_min_area: float = 100.0,
    cyl_min_area: float = 80.0,
    cyl_normal_var_deg: float = 12.0,
):
    """
    Returns:
      planes: list[ScanPlaneFeature]
      cyls: list[ScanCylinderFeature]
      remaining_tri_mask: bool array length n_tri (True means not assigned to any feature)
    """

    scan_mesh = load_stl_mesh(scan_stl)

    mesh = o3d.geometry.TriangleMesh(scan_mesh)  # copy
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    N = np.asarray(mesh.triangle_normals)

    tri_area, tri_cent = _triangle_areas_and_centroids(V, F)
    adj = _build_triangle_adjacency(F)

    assigned = np.zeros(len(F), dtype=bool)

    # ---- planes ----
    planes_raw = _region_grow_planes(mesh, angle_deg=plane_angle_deg, dist_tol=plane_dist_tol, min_area=plane_min_area)
    planes: list[ScanPlaneFeature] = []
    for pid, region, n, d, c, area, rmse in planes_raw:
        assigned[region] = True
        sub = _submesh_from_triangles(mesh, region)
        planes.append(ScanPlaneFeature(
            id=int(pid),
            tri_indices=region.astype(np.int32),
            mesh=sub,
            normal=n.astype(np.float64),
            d=float(d),
            centroid=c.astype(np.float64),
            area=float(area),
            rmse=float(rmse),
        ))

    # ---- cylinders ----
    # candidate triangles: not in planes and with higher normal variation (curved)
    # compute per-triangle normal variation = mean angle to neighbors
    cos_var = np.cos(np.deg2rad(cyl_normal_var_deg))
    candidate = (~assigned).copy()

    # build regions on candidate triangles by normal continuity but NOT too strict (curved surfaces)
    visited = np.zeros(len(F), dtype=bool)
    cyl_regions = []
    for seed in range(len(F)):
        if visited[seed] or not candidate[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        region = []
        while stack:
            t = stack.pop()
            region.append(t)
            nt = _unit(N[t])
            for nb in adj[t]:
                if visited[nb] or not candidate[nb]:
                    continue
                nnb = _unit(N[nb])
                # for cylinder we allow larger angle changes than plane:
                if abs(float(nt @ nnb)) >= cos_var:
                    visited[nb] = True
                    stack.append(nb)
        region = np.array(region, dtype=np.int32)
        if region.size < 80:
            continue
        area = float(tri_area[region].sum())
        if area < cyl_min_area:
            continue
        cyl_regions.append(region)

    cyls: list[ScanCylinderFeature] = []
    cid = 0
    for region in cyl_regions:
        # fit cylinder
        try:
            o, v, r, rmse = _fit_cylinder_from_region(mesh, region)
        except Exception:
            continue

        # basic sanity checks
        if not np.isfinite(rmse) or r < 0.5 or r > 5000:
            continue
        if rmse > 2.0:  # mm, loose gate for MVP
            continue

        assigned[region] = True
        sub = _submesh_from_triangles(mesh, region)
        cyls.append(ScanCylinderFeature(
            id=int(cid),
            tri_indices=region.astype(np.int32),
            mesh=sub,
            axis_origin=o.astype(np.float64),
            axis_dir=_unit(v.astype(np.float64)),
            radius=float(r),
            rmse=float(rmse),
        ))
        cid += 1

    remaining = ~assigned
    return planes, cyls, remaining, scan_mesh


class RegionGrowingExtractor(BaseFeatureExtractor):
    """
    基于法线和区域生长的传统特征提取算法适配器
    """
    def extract(self, scan_stl: str, **kwargs):
        # 1. 安全提取特有参数（带默认值防崩）
        plane_angle_deg = kwargs.get("plane_angle_deg", 6.0)
        plane_dist_tol = kwargs.get("plane_dist_tol", 1.0)
        plane_min_area = kwargs.get("plane_min_area", 100.0)
        cyl_min_area = kwargs.get("cyl_min_area", 80.0)
        cyl_normal_var_deg = kwargs.get("cyl_normal_var_deg", 12.0)

        # 2. 调用上面的核心数学函数
        return extract_scan_planes_and_cylinders_from_mesh(
            scan_stl=scan_stl,
            plane_angle_deg=plane_angle_deg,
            plane_dist_tol=plane_dist_tol,
            plane_min_area=plane_min_area,
            cyl_min_area=cyl_min_area,
            cyl_normal_var_deg=cyl_normal_var_deg
        )