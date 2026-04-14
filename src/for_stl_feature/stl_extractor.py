import numpy as np
import open3d as o3d

# 导入你现有的数据结构
from .core_types import ScanPlaneFeature, ScanCylinderFeature

# 导入现有的基于区域生长的算法
from .scan_features_stl import RegionGrowingExtractor
from .extractor_cgal import CGALExtractor
# from .scan_features_ransac import RansacExtractor  # 未来接入


# ✨ 核心：算法注册表 (字典映射)
EXTRACTOR_REGISTRY = {
    "region_growing": RegionGrowingExtractor,
    "ransac": CGALExtractor,
}


def _is_reasonable_cylinder_feature(cyl: ScanCylinderFeature) -> bool:
    cyl.mesh.compute_triangle_normals()
    normals = np.asarray(cyl.mesh.triangle_normals, dtype=np.float64)
    vertices = np.asarray(cyl.mesh.vertices, dtype=np.float64)

    if len(normals) < 10 or len(vertices) == 0:
        return False
    if not np.isfinite(cyl.radius) or cyl.radius <= 0.0:
        return False

    axis_dir = np.asarray(cyl.axis_dir, dtype=np.float64)
    axis_norm = np.linalg.norm(axis_dir)
    if axis_norm <= 1e-12:
        return False
    axis_dir = axis_dir / axis_norm

    axis_alignment = np.abs(normals @ axis_dir)
    if float(np.median(axis_alignment)) > np.sin(np.deg2rad(25.0)):
        return False

    bbox_extent = vertices.max(axis=0) - vertices.min(axis=0)
    patch_span = float(np.linalg.norm(bbox_extent))
    if patch_span <= 1e-6:
        return False

    if cyl.radius > patch_span * 12.0:
        return False

    radial = normals - np.outer(normals @ axis_dir, axis_dir)
    radial_norm = np.linalg.norm(radial, axis=1)
    valid = radial_norm > 1e-6
    if int(np.count_nonzero(valid)) < 10:
        return False

    radial_dirs = radial[valid] / radial_norm[valid, None]
    curvature_spread = 1.0 - float(np.linalg.norm(radial_dirs.mean(axis=0)))
    if curvature_spread < 0.002 and cyl.radius > patch_span * 4.0:
        return False

    return True


def _cylinder_selection_metrics(
    cyl: ScanCylinderFeature,
) -> tuple[float, float, float, float, float]:
    area = float(cyl.mesh.get_surface_area())
    radius = float(cyl.radius) if np.isfinite(cyl.radius) else 0.0
    vertices = np.asarray(cyl.mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return area, 0.0, radius, 0.0, np.inf

    axis_dir = np.asarray(cyl.axis_dir, dtype=np.float64)
    axis_norm = np.linalg.norm(axis_dir)
    if axis_norm <= 1e-12:
        return area, 0.0, radius, 0.0, np.inf
    axis_dir = axis_dir / axis_norm

    rel = vertices - np.asarray(cyl.axis_origin, dtype=np.float64)
    axial = rel @ axis_dir
    axial_span = float(axial.max() - axial.min()) if len(axial) > 0 else 0.0

    radial = rel - np.outer(axial, axis_dir)
    radial_norm = np.linalg.norm(radial, axis=1)
    valid = radial_norm > 1e-6
    if int(np.count_nonzero(valid)) < 6:
        coverage = 0.0
    else:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(tmp @ axis_dir)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(axis_dir, tmp)
        u = u / (np.linalg.norm(u) + 1e-12)
        w = np.cross(axis_dir, u)
        w = w / (np.linalg.norm(w) + 1e-12)

        radial_dirs = radial[valid] / radial_norm[valid, None]
        ang = np.arctan2(radial_dirs @ w, radial_dirs @ u)
        bins = 48
        idx = np.floor((ang + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
        idx = np.clip(idx, 0, bins - 1)
        hist = np.bincount(idx, minlength=bins)
        coverage = float(np.count_nonzero(hist) / bins)

    rel_rmse = float(cyl.rmse / max(radius, 1.0)) if np.isfinite(cyl.rmse) else np.inf
    return area, axial_span, radius, coverage, rel_rmse


def _rank_cylinders_by_model_relative_score(
    cyls: list[ScanCylinderFeature],
) -> tuple[np.ndarray, np.ndarray]:
    if len(cyls) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 5), dtype=np.float64)

    metrics = np.asarray([_cylinder_selection_metrics(cyl) for cyl in cyls], dtype=np.float64)
    area = np.log1p(np.maximum(metrics[:, 0], 0.0))
    span_ratio = np.log1p(np.maximum(metrics[:, 1], 0.0) / np.maximum(metrics[:, 2], 1e-6))
    coverage = np.clip(metrics[:, 3], 0.0, 1.0)
    quality = np.log1p(1.0 / np.maximum(metrics[:, 4], 1e-6))
    radius = np.log1p(np.maximum(metrics[:, 2], 0.0))
    features = np.stack([area, span_ratio, coverage, quality, radius], axis=1)

    med = np.median(features, axis=0)
    mad = np.median(np.abs(features - med[None, :]), axis=0)
    scaled = (features - med[None, :]) / np.maximum(mad[None, :] * 1.4826, 1e-6)

    weights = np.asarray([1.15, 1.0, 0.35, 0.35, 0.15], dtype=np.float64)
    score = scaled @ weights
    return score.astype(np.float64), metrics


def _filter_dominant_cylinders_threshold(
    cyls: list[ScanCylinderFeature],
    min_area_ratio: float,
    min_span_radius_ratio: float,
    min_keep: int,
) -> list[ScanCylinderFeature]:
    if len(cyls) <= 1:
        return cyls

    metrics = [_cylinder_selection_metrics(cyl) for cyl in cyls]
    top_area = max((m[0] for m in metrics), default=0.0)
    if top_area <= 0.0:
        return cyls

    kept: list[ScanCylinderFeature] = []
    for cyl, (area, axial_span, radius) in zip(cyls, metrics):
        if area < top_area * min_area_ratio:
            continue
        if radius > 1e-6 and axial_span < radius * min_span_radius_ratio:
            continue
        kept.append(cyl)

    if len(kept) >= min_keep:
        return kept

    ranked = sorted(
        zip(cyls, metrics),
        key=lambda item: (
            -item[1][0],
            -(item[1][1] / max(item[1][2], 1e-6)),
            item[0].rmse,
        ),
    )
    return [cyl for cyl, _ in ranked[:max(min_keep, len(kept), 1)]]


def _filter_dominant_cylinders_cluster(
    cyls: list[ScanCylinderFeature],
    min_keep: int,
) -> list[ScanCylinderFeature]:
    if len(cyls) <= 2:
        return cyls

    score, metrics = _rank_cylinders_by_model_relative_score(cyls)
    if len(score) == 0:
        return cyls

    ranked_idx = np.argsort(-score)
    if len(ranked_idx) <= min_keep:
        return [cyls[idx] for idx in ranked_idx.tolist()]

    ranked_score = score[ranked_idx]
    gaps = ranked_score[:-1] - ranked_score[1:]
    search_start = max(min_keep - 1, 0)
    search_end = min(len(gaps), max(search_start + 1, int(np.ceil(len(ranked_idx) * 0.6))))

    if search_start >= search_end:
        keep_count = min(len(ranked_idx), max(min_keep, 1))
    else:
        local_gap_idx = int(np.argmax(gaps[search_start:search_end]))
        keep_count = search_start + local_gap_idx + 1

    keep_count = min(len(ranked_idx), max(keep_count, min_keep))
    keep_idx = ranked_idx[:keep_count]

    dominant_median_area = float(np.median(metrics[keep_idx, 0])) if keep_idx.size else 0.0
    trailing_area = float(np.median(metrics[ranked_idx[keep_count:], 0])) if keep_count < len(ranked_idx) else 0.0
    if trailing_area > dominant_median_area and keep_count < len(ranked_idx):
        keep_count = min(len(ranked_idx), max(min_keep, int(np.ceil(len(ranked_idx) * 0.35))))
        keep_idx = ranked_idx[:keep_count]

    return [cyls[idx] for idx in keep_idx.tolist()]


def process_scan_features(
    scan_stl: str, 
    method: str = "region_growing",  # 默认使用区域生长
    max_planes: int = 60, 
    max_cyls: int = 40,
    **kwargs
):
    print(f"[Pipeline] 开始处理 STL: {scan_stl} | 引擎: {method}")

    # 1. 动态获取算法类
    extractor_class = EXTRACTOR_REGISTRY.get(method)
    if not extractor_class:
        raise ValueError(f"未知的算法: {method}。目前支持: {list(EXTRACTOR_REGISTRY.keys())}")

    # 2. 实例化算法并提取特征 (多态调用)
    extractor = extractor_class()
    scan_planes_m, scan_cyls_m, remaining_mask, scan_mesh = extractor.extract(scan_stl, **kwargs)

    # 3. 拦截异常
    if not scan_planes_m and not scan_cyls_m:
        raise RuntimeError(f"使用 {method} 算法未在 STL 中提取到任何特征: {scan_stl}")

    # 4. 二次校验 (剔除球面伪圆柱)
    valid_cyls = []
    for cyl in scan_cyls_m:
        if _is_reasonable_cylinder_feature(cyl):
            valid_cyls.append(cyl)

    cyl_selection_mode = str(kwargs.get("cyl_selection_mode", "none")).lower()
    if cyl_selection_mode == "none" and bool(kwargs.get("cyl_dominant_only", False)):
        cyl_selection_mode = "dominant_threshold"

    cyl_dominant_min_keep = int(kwargs.get("cyl_dominant_min_keep", min(max_cyls, 8)))
    if cyl_selection_mode == "dominant_threshold":
        valid_cyls = _filter_dominant_cylinders_threshold(
            cyls=valid_cyls,
            min_area_ratio=float(kwargs.get("cyl_dominant_min_area_ratio", 0.05)),
            min_span_radius_ratio=float(kwargs.get("cyl_dominant_min_span_radius_ratio", 0.25)),
            min_keep=cyl_dominant_min_keep,
        )
    elif cyl_selection_mode == "dominant_cluster":
        valid_cyls = _filter_dominant_cylinders_cluster(
            cyls=valid_cyls,
            min_keep=cyl_dominant_min_keep,
        )

    # 5. 综合排序与截断
    final_planes = sorted(scan_planes_m, key=lambda p: p.area, reverse=True)[:max_planes]
    cyl_sort_mode = str(kwargs.get("cyl_sort_mode", "area_first")).lower()
    if cyl_sort_mode == "rmse_first":
        final_cyls = sorted(valid_cyls, key=lambda c: (c.rmse, -c.mesh.get_surface_area()))[:max_cyls]
    else:
        final_cyls = sorted(valid_cyls, key=lambda c: (-c.mesh.get_surface_area(), c.rmse))[:max_cyls]

    print(f"[Pipeline] 提取完成: 发现 {len(final_planes)} 个平面, {len(final_cyls)} 个有效圆柱.")
    return final_planes, final_cyls, remaining_mask, scan_mesh
