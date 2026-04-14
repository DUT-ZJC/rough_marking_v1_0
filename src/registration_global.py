import numpy as np
import open3d as o3d

from .logging_utils import log


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    return np.einsum("kj,ij->ki", points, R) + t[None, :]


def _pick_nearest_feasible_shift(low: float, high: float) -> tuple[float, bool]:
    if low <= high:
        if low <= 0.0 <= high:
            return 0.0, True
        if high < 0.0:
            return float(high), True
        return float(low), True
    return float(0.5 * (low + high)), False


def _compute_obb_enclosure_shift(
    cad_vertices: np.ndarray,
    T_cad_to_scan: np.ndarray,
    envelope_mesh: o3d.geometry.TriangleMesh,
    margin: float,
) -> tuple[np.ndarray, dict]:
    detail = {
        "obb_feasible": True,
        "obb_local_shift": [0.0, 0.0, 0.0],
        "obb_world_shift_norm": 0.0,
    }

    cad_vertices = np.asarray(cad_vertices, dtype=np.float64)
    if cad_vertices.size == 0:
        return np.zeros(3, dtype=np.float64), detail

    try:
        obb = envelope_mesh.get_oriented_bounding_box(robust=False)
    except Exception:
        obb = envelope_mesh.get_oriented_bounding_box(robust=True)
    R_box = np.asarray(obb.R, dtype=np.float64)
    center = np.asarray(obb.center, dtype=np.float64)
    half_extent = 0.5 * np.asarray(obb.extent, dtype=np.float64)

    cad_scan = _transform_points(cad_vertices, T_cad_to_scan)
    cad_local = np.einsum("kj,ji->ki", cad_scan - center, R_box)

    cad_min = cad_local.min(axis=0)
    cad_max = cad_local.max(axis=0)
    lower = (-half_extent + margin) - cad_min
    upper = (half_extent - margin) - cad_max

    shift_local = np.zeros(3, dtype=np.float64)
    feasible = True
    for axis in range(3):
        shift_local[axis], axis_ok = _pick_nearest_feasible_shift(
            float(lower[axis]),
            float(upper[axis]),
        )
        feasible = feasible and axis_ok

    shift_world = np.einsum("ij,j->i", R_box, shift_local)
    detail["obb_feasible"] = bool(feasible)
    detail["obb_local_shift"] = shift_local.tolist()
    detail["obb_world_shift_norm"] = float(np.linalg.norm(shift_world))
    return shift_world, detail


def _build_enclosure_mesh(scan_mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, str]:
    if scan_mesh.is_watertight():
        return scan_mesh, "scan_mesh"

    log("[CoarseEnclose] scan mesh is not watertight, using convex hull as enclosure mesh.")
    hull_mesh, _ = scan_mesh.compute_convex_hull()
    if not hull_mesh.has_triangle_normals():
        hull_mesh.compute_triangle_normals()
    if not hull_mesh.has_vertex_normals():
        hull_mesh.compute_vertex_normals()
    return hull_mesh, "convex_hull"


def _refine_translation_for_enclosure(
    cad_surface_points: np.ndarray,
    T_cad_to_scan: np.ndarray,
    enclosure_mesh: o3d.geometry.TriangleMesh,
    margin: float,
    max_iters: int,
    outside_tol: float = 1e-4,
) -> tuple[np.ndarray, dict]:
    detail = {
        "iterations": 0,
        "outside_before": 0,
        "outside_after": 0,
        "max_signed_distance_before": 0.0,
        "max_signed_distance_after": 0.0,
        "min_signed_distance_after": 0.0,
        "iter_shift_norm": 0.0,
    }

    cad_surface_points = np.asarray(cad_surface_points, dtype=np.float64)
    if cad_surface_points.size == 0:
        return np.asarray(T_cad_to_scan, dtype=np.float64).copy(), detail

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(enclosure_mesh))

    T = np.asarray(T_cad_to_scan, dtype=np.float64).copy()
    total_shift = np.zeros(3, dtype=np.float64)
    outside_before = None
    max_sd_before = None

    for it in range(max_iters + 1):
        pts_scan = _transform_points(cad_surface_points, T).astype(np.float32)
        tensor = o3d.core.Tensor(pts_scan, dtype=o3d.core.Dtype.Float32)
        sd = scene.compute_signed_distance(tensor).numpy().astype(np.float64)
        residual = sd + margin
        outside = residual > outside_tol

        if outside_before is None:
            outside_before = int(np.count_nonzero(outside))
            max_sd_before = float(sd.max()) if len(sd) > 0 else 0.0

        if not np.any(outside):
            detail["iterations"] = int(it)
            break

        if it >= max_iters:
            detail["iterations"] = int(max_iters)
            break

        outside_pts = pts_scan[outside]
        outside_tensor = o3d.core.Tensor(outside_pts, dtype=o3d.core.Dtype.Float32)
        closest = scene.compute_closest_points(outside_tensor)["points"].numpy().astype(np.float64)
        deltas = closest - outside_pts.astype(np.float64)

        weights = residual[outside]
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-12:
            step = deltas.mean(axis=0)
        else:
            step = (deltas * weights[:, None]).sum(axis=0) / weight_sum

        step_norm = float(np.linalg.norm(step))
        T[:3, 3] += step
        total_shift += step

        if step_norm <= 1e-6:
            detail["iterations"] = int(it + 1)
            break
    else:
        detail["iterations"] = int(max_iters)

    pts_final = _transform_points(cad_surface_points, T).astype(np.float32)
    sd_final = scene.compute_signed_distance(
        o3d.core.Tensor(pts_final, dtype=o3d.core.Dtype.Float32)
    ).numpy().astype(np.float64)
    residual_final = sd_final + margin
    outside_final = residual_final > outside_tol

    detail["outside_before"] = int(0 if outside_before is None else outside_before)
    detail["outside_after"] = int(np.count_nonzero(outside_final))
    detail["max_signed_distance_before"] = float(0.0 if max_sd_before is None else max_sd_before)
    detail["max_signed_distance_after"] = float(sd_final.max()) if len(sd_final) > 0 else 0.0
    detail["min_signed_distance_after"] = float(sd_final.min()) if len(sd_final) > 0 else 0.0
    detail["iter_shift_norm"] = float(np.linalg.norm(total_shift))
    return T, detail


def enforce_scan_encloses_cad(
    scan_mesh: o3d.geometry.TriangleMesh,
    cad_mesh: o3d.geometry.TriangleMesh,
    T_cad_to_scan: np.ndarray,
    margin: float = 0.0,
    sample_points: int = 12000,
    max_iters: int = 8,
) -> tuple[np.ndarray, dict]:
    detail = {
        "enabled": True,
        "margin": float(margin),
        "sample_points": int(sample_points),
        "enclosure_mesh": "scan_mesh",
        "obb_feasible": True,
        "obb_local_shift": [0.0, 0.0, 0.0],
        "obb_world_shift_norm": 0.0,
        "iterations": 0,
        "outside_before": 0,
        "outside_after": 0,
        "max_signed_distance_before": 0.0,
        "max_signed_distance_after": 0.0,
        "min_signed_distance_after": 0.0,
        "iter_shift_norm": 0.0,
        "status": "skipped",
    }

    if len(cad_mesh.vertices) == 0 or len(scan_mesh.vertices) == 0:
        detail["status"] = "skipped_empty_mesh"
        return np.asarray(T_cad_to_scan, dtype=np.float64).copy(), detail

    T = np.asarray(T_cad_to_scan, dtype=np.float64).copy()
    cad_vertices = np.asarray(cad_mesh.vertices, dtype=np.float64)

    enclosure_mesh, enclosure_kind = _build_enclosure_mesh(scan_mesh)
    detail["enclosure_mesh"] = enclosure_kind

    obb_shift, obb_detail = _compute_obb_enclosure_shift(
        cad_vertices=cad_vertices,
        T_cad_to_scan=T,
        envelope_mesh=enclosure_mesh,
        margin=margin,
    )
    T[:3, 3] += obb_shift
    detail.update(obb_detail)

    n_surface = max(1024, int(sample_points))
    cad_surface_points = np.asarray(
        cad_mesh.sample_points_uniformly(number_of_points=n_surface).points,
        dtype=np.float64,
    )

    T, refine_detail = _refine_translation_for_enclosure(
        cad_surface_points=cad_surface_points,
        T_cad_to_scan=T,
        enclosure_mesh=enclosure_mesh,
        margin=margin,
        max_iters=max_iters,
    )
    detail.update(refine_detail)
    detail["status"] = "ok" if detail["outside_after"] == 0 else "partial"
    return T, detail


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel: float, normal_radius: float) -> o3d.geometry.PointCloud:
    log(f"Downsample voxel={voxel}")
    pcd_ds = pcd.voxel_down_sample(voxel)
    log(f"Estimate normals radius={normal_radius}")
    pcd_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    pcd_ds.orient_normals_consistent_tangent_plane(30)
    return pcd_ds


def _prep(pcd, voxel):
    p = pcd.voxel_down_sample(voxel)
    p.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    f = o3d.pipelines.registration.compute_fpfh_feature(
        p,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
    )
    return p, f


def _icp_refine(src, tgt, T, dist, iters=60):
    if not tgt.has_normals():
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=dist * 2.0, max_nn=30)
        )
    return o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        dist,
        T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),
    )


def stable_coarse_register(
    src_full: o3d.geometry.PointCloud,
    tgt_full: o3d.geometry.PointCloud,
    voxel_candidates=(12.0, 10.0, 8.0),
    refine_voxels=(6.0, 4.0, 2.0),
    num_trials=5,
    lambda_rmse=0.10,
):
    best = None
    best_score = -1e18
    best_detail = {}

    for v0 in voxel_candidates:
        src0, f0 = _prep(src_full, v0)
        tgt0, g0 = _prep(tgt_full, v0)

        dist0 = v0 * 2.0
        log(f"[StableCoarse] voxel={v0} dist0={dist0:.2f} trials={num_trials}")

        for t in range(num_trials):
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src0,
                tgt0,
                f0,
                g0,
                mutual_filter=False,
                max_correspondence_distance=dist0,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist0),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(120000, 500),
            )

            T = ransac.transformation

            for vr in refine_voxels:
                src_d = src_full.voxel_down_sample(vr)
                tgt_d = tgt_full.voxel_down_sample(vr)
                dist = vr * 2.0
                icp = _icp_refine(src_d, tgt_d, T, dist, iters=60)
                T = icp.transformation

            eval_res = o3d.pipelines.registration.evaluate_registration(
                src_full,
                tgt_full,
                max_correspondence_distance=4.0,
                transformation=T,
            )
            fitness = float(eval_res.fitness)
            rmse = float(eval_res.inlier_rmse)
            score = fitness - lambda_rmse * rmse

            if score > best_score:
                best_score = score
                best = T
                best_detail = {
                    "voxel0": v0,
                    "trial": t,
                    "fitness": fitness,
                    "rmse": rmse,
                    "score": score,
                }

            log(f"  trial={t} eval fitness={fitness:.4f} rmse={rmse:.4f} score={score:.4f}")

    if best is None:
        raise RuntimeError("Stable coarse registration failed (no result).")

    log(f"[StableCoarse] BEST: {best_detail}")
    return best, best_detail


def registration_coarse(scan_mesh, cad_mesh, config):
    scan_pcd_raw = scan_mesh.sample_points_uniformly(number_of_points=300000)
    cad_pcd_raw = cad_mesh.sample_points_uniformly(number_of_points=25000)

    cad_pcd = preprocess_pcd(
        cad_pcd_raw,
        voxel=config.VOXEL_SIZE,
        normal_radius=config.NORMAL_RADIUS,
    )
    scan_pcd = preprocess_pcd(
        scan_pcd_raw,
        voxel=config.VOXEL_SIZE,
        normal_radius=config.NORMAL_RADIUS,
    )

    T_coarse, info = stable_coarse_register(
        cad_pcd,
        scan_pcd,
        voxel_candidates=(12.0, 10.0, 8.0),
        refine_voxels=(6.0, 4.0, 2.0),
        num_trials=6,
        lambda_rmse=0.10,
    )

    coarse_info = dict(info)
    if getattr(config, "COARSE_ENCLOSE_ENABLE", True):
        T_coarse, enclosure_info = enforce_scan_encloses_cad(
            scan_mesh=scan_mesh,
            cad_mesh=cad_mesh,
            T_cad_to_scan=T_coarse,
            margin=float(getattr(config, "COARSE_ENCLOSE_MARGIN", 0.0)),
            sample_points=int(getattr(config, "COARSE_ENCLOSE_SAMPLE_POINTS", 12000)),
            max_iters=int(getattr(config, "COARSE_ENCLOSE_MAX_ITERS", 8)),
        )
        coarse_info["enclosure"] = enclosure_info
        log(f"[CoarseEnclose] info: {enclosure_info}")
    else:
        coarse_info["enclosure"] = {"enabled": False, "status": "disabled"}

    return T_coarse, coarse_info
