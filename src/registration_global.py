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
#估计法向

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
#icp

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
    #点云化

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
    #点云法向预处理

    T_coarse, info = stable_coarse_register(
        cad_pcd,
        scan_pcd,
        voxel_candidates=(12.0, 10.0, 8.0),
        refine_voxels=(6.0, 4.0, 2.0),
        num_trials=6,
        lambda_rmse=0.10,
    )
    #点云粗配准

    return T_coarse, info
