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

    # 5. 综合排序与截断
    final_planes = sorted(scan_planes_m, key=lambda p: p.area, reverse=True)[:max_planes]
    final_cyls = sorted(valid_cyls, key=lambda c: (c.rmse, -c.mesh.get_surface_area()))[:max_cyls]

    print(f"[Pipeline] 提取完成: 发现 {len(final_planes)} 个平面, {len(final_cyls)} 个有效圆柱.")
    return final_planes, final_cyls, remaining_mask, scan_mesh
