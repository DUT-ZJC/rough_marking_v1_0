from __future__ import annotations

import os
import numpy as np
import config
from src.logging_utils import log, validate_feature_triangles

from src.for_step_feature_fit import (
    FitThresholds as StepFitThresholds,
    RegistrationConfig as StepFitRegistrationConfig,
    process_scan_features_step_guided,
)

from src.cad_features_step import extract_cad_planes_and_cylinders

from src.registration_global import registration_coarse

from src.viewer_dual_pick import DualPickerApp, PairConstraintSpec
# NEW: scan mesh analytic-like features
from src.visualize import show_alignment_mesh

# 优化对导入
from src.feature_objective import (
    build_feature_terms,
    RigidFeatureOptimizationProblem,
    RigidFeatureOptimizer,
)
#评估模块导入
from src.feature_evaluation import (
    evaluate_plane_plane_terms,
    evaluate_cyl_plane_terms,
    evaluate_cyl_cyl_terms,
    format_distance_stats,
    format_cylinder_stats,
)

import faulthandler; faulthandler.enable()




def main():

    # ==== 0) 定位地址 =====

    cad_step = config.CAD_STEP_PATH
    scan_stl = config.SCAN_STL_PATH

    if not os.path.exists(cad_step):
        raise FileNotFoundError(f"CAD STEP not found: {cad_step}")
    if not os.path.exists(scan_stl):
        raise FileNotFoundError(f"Scan STL not found: {scan_stl}")

    # ===== 1) 找特征并加载open3d网格 =====

    # cad的特征获取
    planes, cyls, _unknown,cad_base_mesh = extract_cad_planes_and_cylinders(cad_step, linear_deflection=0.5)
    if not planes and not cyls:
        raise RuntimeError("No analytic plane/cylinder features found in STEP.")
    
    # 毛坯特征提取
    backend = str(getattr(config, "SCAN_FEATURE_BACKEND", "step_guided")).strip().lower()
    if backend not in {"", "step_guided"}:
        raise ValueError(
            f"Unsupported SCAN_FEATURE_BACKEND={backend!r}. "
            "Only 'step_guided' is supported now."
        )
    log("Using STEP-guided scan feature backend.")
    scan_planes_m, scan_cyls_m, remaining_mask, scan_mesh = process_scan_features_step_guided(
        step_path=cad_step,
        scan_stl_path=scan_stl,
        linear_deflection=float(getattr(config, "STEP_GUIDED_LINEAR_DEFLECTION", 0.5)),
        thresholds=StepFitThresholds(
            support_gap_mm=float(getattr(config, "STEP_GUIDED_SUPPORT_GAP_MM", 2.0)),
            plane_tol_mm=float(getattr(config, "STEP_GUIDED_PLANE_TOL_MM", 1.0)),
            cylinder_tol_mm=float(getattr(config, "STEP_GUIDED_CYLINDER_TOL_MM", 1.0)),
            generic_tol_mm=float(getattr(config, "STEP_GUIDED_GENERIC_TOL_MM", 1.5)),
            min_support_triangles=int(getattr(config, "STEP_GUIDED_MIN_SUPPORT_TRIANGLES", 20)),
        ),
        registration_config=StepFitRegistrationConfig(
            VOXEL_SIZE=float(getattr(config, "VOXEL_SIZE", 2.0)),
            NORMAL_RADIUS=float(getattr(config, "NORMAL_RADIUS", 6.0)),
            COARSE_ENCLOSE_ENABLE=bool(getattr(config, "COARSE_ENCLOSE_ENABLE", True)),
            COARSE_ENCLOSE_MARGIN=float(getattr(config, "COARSE_ENCLOSE_MARGIN", 0.0)),
            COARSE_ENCLOSE_SAMPLE_POINTS=int(getattr(config, "COARSE_ENCLOSE_SAMPLE_POINTS", 12000)),
            COARSE_ENCLOSE_MAX_ITERS=int(getattr(config, "COARSE_ENCLOSE_MAX_ITERS", 8)),
        ),
        use_global_results=bool(getattr(config, "STEP_GUIDED_USE_GLOBAL_RESULTS", True)),
    )


    #获取特征的三角面片id的集合

    # ===== 2) 粗配准 =====

    T_coarse, info = registration_coarse(
        scan_mesh=scan_mesh,
        cad_mesh=cad_base_mesh,
        config=config,
    )

    log(f"Stable coarse info: {info}")

    log("About to visualize COARSE alignment (close window to continue)...")
    show_alignment_mesh(cad_base_mesh, scan_mesh, T_coarse)

    """
    这里结束了初始配准；

    得到的操作对象：
    cad_base_mesh
    scan_mesh

    """

    # ===== 3) 选择约束对 =====

    #检查特征是否存在

    validate_feature_triangles(cad_base_mesh, planes, "cad_plane")
    validate_feature_triangles(cad_base_mesh, cyls, "cad_cyl")
    validate_feature_triangles(scan_mesh, scan_planes_m, "scan_plane")
    validate_feature_triangles(scan_mesh, scan_cyls_m, "scan_cyl")



    picker = DualPickerApp(
        cad_plane_features=planes,
        cad_cyl_features=cyls,
        scan_plane_features=scan_planes_m,
        scan_cyl_features=scan_cyls_m,
        cad_base_mesh=cad_base_mesh,
        scan_base_mesh=scan_mesh,
    )
    #人机交互选中特征

    #收集约束列表  
    pair_constraints: list[PairConstraintSpec] = picker.run()

    if not pair_constraints:
        log("No constraints specified. Exit.")
        return

    log(f"Paired constraints confirmed: {len(pair_constraints)}")
    for c in pair_constraints:
        log(f"  {c}")


    # ===== 4) 建立目标函数 =====

    plane_plane_terms, cyl_cyl_terms, cyl_plane_terms = build_feature_terms(
        pair_constraints=pair_constraints,
        cad_planes=planes,
        cad_cyls=cyls,
        scan_planes=scan_planes_m,
        scan_cyls=scan_cyls_m,
        max_plane_points=1000,   # 可以先给个默认值，后面再调
    )
    #构造objective

    log(f"plane-plane terms: {len(plane_plane_terms)}")
    log(f"cyl-cyl terms: {len(cyl_cyl_terms)}")
    log(f"cyl-plane terms: {len(cyl_plane_terms)}")

    R0 = T_coarse[:3, :3].copy()
    t0 = T_coarse[:3, 3].copy()
    #粗配准初值


    problem = RigidFeatureOptimizationProblem(
        R0=R0,
        t0=t0,
        rot_bound_deg=np.array([5.0, 5.0, 5.0], dtype=np.float64),  # 举例：围绕粗配准初值 ±5°
        plane_plane_terms=plane_plane_terms,
        cyl_cyl_terms=cyl_cyl_terms,
        cyl_plane_terms=cyl_plane_terms,
    )
    #构造问题

    optimizer = RigidFeatureOptimizer(problem)
    res = optimizer.solve()

    if not res["ok"]:
        raise RuntimeError(f"Optimization failed: {res['message']}")

    T_final = res["T"]
    log("==== Optimization Output ====")
    log(f"Final transform T (CAD->SCAN):\n{T_final}")
    log(f"Final cost: {res['cost']:.6f}, nfev={res['nfev']}")
    #优化

    log("About to visualize FINAL alignment...")
    show_alignment_mesh(cad_base_mesh, scan_mesh, T_final)

    # ===== 5）评估 =====
    R_eval = T_final[:3, :3]
    t_eval = T_final[:3, 3]

    pp_evals = evaluate_plane_plane_terms(plane_plane_terms, R=R_eval, t=t_eval)
    for ev in pp_evals:
        log(format_distance_stats(
            f"[Plane-Plane] CAD#{ev.cad_id} vs SCAN#{ev.scan_id}",
            ev.stats
        ))

    cp_evals = evaluate_cyl_plane_terms(cyl_plane_terms, R=R_eval, t=t_eval)
    for ev in cp_evals:
        log(format_distance_stats(
            f"[Cyl-Plane] CAD#{ev.cad_id} vs SCAN#{ev.scan_id}",
            ev.stats
        ))

    scan_cyl_map = {c.id: c for c in scan_cyls_m}
    cc_evals = evaluate_cyl_cyl_terms(cyl_cyl_terms, scan_cyl_map=scan_cyl_map, R=R_eval, t=t_eval)
    for ev in cc_evals:
        log(format_cylinder_stats(
            f"[Cyl-Cyl] CAD#{ev.cad_id} vs SCAN#{ev.scan_id}",
            ev.stats
        ))


if __name__ == "__main__":
    main()
