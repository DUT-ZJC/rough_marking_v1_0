from __future__ import annotations

"""
feature_evaluation.py

用于评估特征 patch 上所有采样点到“基准几何”的距离统计：
- 最远距离（max）
- 最近距离（min）
- 平均距离（mean）
- 与目标距离 d 的对比误差

设计目标
--------
1. 与 feature_objective.py 的建模保持一致；
2. 既可用于优化前评估，也可用于优化后评估；
3. 优先使用“特征 patch 上的三角面中心”作为样本点；
4. 输出结构清晰，便于 main.py 直接打印、保存或进一步分析。
"""

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np
import open3d as o3d


# ============================================================
# 基础工具
# ============================================================

def unit(v: np.ndarray) -> np.ndarray:
    """单位化向量。"""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def triangle_centers_from_mesh(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """计算三角网格所有三角面的中心点。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    if len(vertices) == 0 or len(triangles) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    centers = (
        vertices[triangles[:, 0]]
        + vertices[triangles[:, 1]]
        + vertices[triangles[:, 2]]
    ) / 3.0
    return centers


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    对点集施加刚体变换：
        p' = R p + t
    """
    points = np.asarray(points, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return (R @ points.T).T + t[None, :]


# ============================================================
# 结果数据结构
# ============================================================

@dataclass
class DistanceStats:
    """
    平面类/轴线到平面的距离统计结果。

    signed_*:
        有符号距离统计
    abs_*:
        绝对距离统计
    target_d:
        当时约束里定义的目标距离 d
    err_to_target_*:
        与目标距离 d 的偏差统计
    """
    count: int

    signed_min: float
    signed_max: float
    signed_mean: float

    abs_min: float
    abs_max: float
    abs_mean: float

    target_d: float

    err_to_target_min: float
    err_to_target_max: float
    err_to_target_mean: float
    err_to_target_abs_mean: float


@dataclass
class PlanePlaneEvaluation:
    """Scan 平面 patch 上所有样本点到变换后 CAD 解析平面的距离统计。"""
    cad_id: int
    scan_id: int
    stats: DistanceStats


@dataclass
class CylPlaneEvaluation:
    """CAD 圆柱轴线上采样点到 Scan 平面的距离统计。"""
    cad_id: int
    scan_id: int
    stats: DistanceStats


@dataclass
class CylinderRadialStats:
    """
    点到圆柱解析面的径向距离统计。

    radial_*:
        点到轴线的径向距离统计
    target_radius:
        解析圆柱半径
    err_to_radius_*:
        与目标半径的偏差统计
    """
    count: int

    radial_min: float
    radial_max: float
    radial_mean: float

    target_radius: float

    err_to_radius_min: float
    err_to_radius_max: float
    err_to_radius_mean: float
    err_to_radius_abs_mean: float


@dataclass
class CylCylEvaluation:
    """Scan 圆柱 patch 上所有样本点到变换后 CAD 解析圆柱面的径向评估。"""
    cad_id: int
    scan_id: int
    stats: CylinderRadialStats


# ============================================================
# 解析几何距离
# ============================================================

def signed_point_to_plane_distance(points: np.ndarray, n: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    点到平面的有符号距离：
        d(p, Π) = n^T (p - q)
    """
    points = np.asarray(points, dtype=np.float64)
    n = unit(n)
    q = np.asarray(q, dtype=np.float64).reshape(3)

    if points.size == 0:
        return np.zeros((0,), dtype=np.float64)

    return (points - q[None, :]) @ n


def point_to_axis_radial_distance(points: np.ndarray, o: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    点到轴线的径向距离：
        || (I - v v^T) (p - o) ||
    """
    points = np.asarray(points, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64).reshape(3)
    v = unit(v)

    if points.size == 0:
        return np.zeros((0,), dtype=np.float64)

    proj = np.eye(3, dtype=np.float64) - np.outer(v, v)
    diff = points - o[None, :]
    perp = (proj @ diff.T).T
    return np.linalg.norm(perp, axis=1)


# ============================================================
# 统计工具
# ============================================================

def _distance_stats(values: np.ndarray, target_d: float) -> DistanceStats:
    """
    给定一组“有符号距离” values，计算：
    - 有符号 min/max/mean
    - 绝对值 min/max/mean
    - 与目标距离 d 的偏差统计
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    target_d = float(target_d)

    if values.size == 0:
        return DistanceStats(
            count=0,
            signed_min=np.nan,
            signed_max=np.nan,
            signed_mean=np.nan,
            abs_min=np.nan,
            abs_max=np.nan,
            abs_mean=np.nan,
            target_d=target_d,
            err_to_target_min=np.nan,
            err_to_target_max=np.nan,
            err_to_target_mean=np.nan,
            err_to_target_abs_mean=np.nan,
        )

    err = values - target_d
    abs_values = np.abs(values)

    return DistanceStats(
        count=int(values.size),
        signed_min=float(np.min(values)),
        signed_max=float(np.max(values)),
        signed_mean=float(np.mean(values)),
        abs_min=float(np.min(abs_values)),
        abs_max=float(np.max(abs_values)),
        abs_mean=float(np.mean(abs_values)),
        target_d=target_d,
        err_to_target_min=float(np.min(err)),
        err_to_target_max=float(np.max(err)),
        err_to_target_mean=float(np.mean(err)),
        err_to_target_abs_mean=float(np.mean(np.abs(err))),
    )


def _cylinder_radial_stats(radial_values: np.ndarray, target_radius: float) -> CylinderRadialStats:
    """
    对点到轴线的径向距离做统计，并与目标半径对比。
    """
    radial_values = np.asarray(radial_values, dtype=np.float64).reshape(-1)
    target_radius = float(target_radius)

    if radial_values.size == 0:
        return CylinderRadialStats(
            count=0,
            radial_min=np.nan,
            radial_max=np.nan,
            radial_mean=np.nan,
            target_radius=target_radius,
            err_to_radius_min=np.nan,
            err_to_radius_max=np.nan,
            err_to_radius_mean=np.nan,
            err_to_radius_abs_mean=np.nan,
        )

    err = radial_values - target_radius

    return CylinderRadialStats(
        count=int(radial_values.size),
        radial_min=float(np.min(radial_values)),
        radial_max=float(np.max(radial_values)),
        radial_mean=float(np.mean(radial_values)),
        target_radius=target_radius,
        err_to_radius_min=float(np.min(err)),
        err_to_radius_max=float(np.max(err)),
        err_to_radius_mean=float(np.mean(err)),
        err_to_radius_abs_mean=float(np.mean(np.abs(err))),
    )


# ============================================================
# 单项评估函数
# ============================================================

def evaluate_plane_plane_term(
    *,
    cad_id: int,
    scan_id: int,
    n_cad: np.ndarray,
    q_cad: np.ndarray,
    scan_points: np.ndarray,
    target_gap_mm: float,
    R: np.ndarray,
    t: np.ndarray,
) -> PlanePlaneEvaluation:
    """
    评估 plane-plane 约束：
    计算 Scan 平面 patch 上所有点到“变换后 CAD 解析平面”的距离统计，
    并与目标 d = target_gap_mm 做对比。

    数学形式
    --------
    CAD 平面变换后：
        n' = R n_cad
        q' = R q_cad + t

    点到平面的有符号距离：
        d_i = n'^T (p_i - q')

    目标距离为：
        target_gap_mm
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    n_cad_p = unit(R @ np.asarray(n_cad, dtype=np.float64).reshape(3))
    q_cad_p = R @ np.asarray(q_cad, dtype=np.float64).reshape(3) + t

    distances = signed_point_to_plane_distance(scan_points, n_cad_p, q_cad_p)
    stats = _distance_stats(distances, target_gap_mm)

    return PlanePlaneEvaluation(
        cad_id=int(cad_id),
        scan_id=int(scan_id),
        stats=stats,
    )


def evaluate_cyl_plane_term(
    *,
    cad_id: int,
    scan_id: int,
    o_cad: np.ndarray,
    v_cad: np.ndarray,
    n_scan: np.ndarray,
    q_scan: np.ndarray,
    target_axis_plane_dist_mm: float,
    R: np.ndarray,
    t: np.ndarray,
    axis_sample_half_length: float = 50.0,
    axis_sample_count: int = 101,
) -> CylPlaneEvaluation:
    """
    评估 cyl-plane 约束：
    在“变换后 CAD 圆柱轴线”上均匀采样若干点，
    统计这些点到 Scan 平面的距离，并与目标距离 d 对比。

    说明
    ----
    对于理想的“轴线平行于平面”情况，轴线上任一点到平面的距离都一样；
    这里仍用多点采样，主要是为了让输出统计格式统一且更直观。
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    o_p = R @ np.asarray(o_cad, dtype=np.float64).reshape(3) + t
    v_p = unit(R @ np.asarray(v_cad, dtype=np.float64).reshape(3))

    ss = np.linspace(-axis_sample_half_length, axis_sample_half_length, int(axis_sample_count), dtype=np.float64)
    axis_points = o_p[None, :] + ss[:, None] * v_p[None, :]

    distances = signed_point_to_plane_distance(
        axis_points,
        np.asarray(n_scan, dtype=np.float64),
        np.asarray(q_scan, dtype=np.float64),
    )
    stats = _distance_stats(distances, target_axis_plane_dist_mm)

    return CylPlaneEvaluation(
        cad_id=int(cad_id),
        scan_id=int(scan_id),
        stats=stats,
    )


def evaluate_cyl_cyl_term(
    *,
    cad_id: int,
    scan_id: int,
    o_cad: np.ndarray,
    v_cad: np.ndarray,
    r_cad: float,
    scan_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> CylCylEvaluation:
    """
    评估 cyl-cyl 约束：
    计算 Scan 圆柱 patch 上所有点到“变换后 CAD 解析圆柱轴线”的径向距离，
    再与 CAD 圆柱半径 r_cad 做对比。

    数学形式
    --------
    CAD 轴线变换后：
        o' = R o_cad + t
        v' = R v_cad

    点到轴线的径向距离：
        rho_i = || (I - v'v'^T)(p_i - o') ||

    与解析圆柱面的偏差：
        rho_i - r_cad
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    o_p = R @ np.asarray(o_cad, dtype=np.float64).reshape(3) + t
    v_p = unit(R @ np.asarray(v_cad, dtype=np.float64).reshape(3))

    radial_values = point_to_axis_radial_distance(scan_points, o_p, v_p)
    stats = _cylinder_radial_stats(radial_values, float(r_cad))

    return CylCylEvaluation(
        cad_id=int(cad_id),
        scan_id=int(scan_id),
        stats=stats,
    )


# ============================================================
# 批量评估函数（直接接 feature_objective 的 term）
# ============================================================

def evaluate_plane_plane_terms(terms: Iterable, R: np.ndarray, t: np.ndarray) -> list[PlanePlaneEvaluation]:
    """批量评估 PlanePlaneTerm 列表。"""
    out: list[PlanePlaneEvaluation] = []
    for term in terms:
        out.append(
            evaluate_plane_plane_term(
                cad_id=term.cad_id,
                scan_id=term.scan_id,
                n_cad=term.n_cad,
                q_cad=term.q_cad,
                scan_points=term.scan_points,
                target_gap_mm=term.target_gap_mm,
                R=R,
                t=t,
            )
        )
    return out


def evaluate_cyl_plane_terms(
    terms: Iterable,
    R: np.ndarray,
    t: np.ndarray,
    axis_sample_half_length: float = 50.0,
    axis_sample_count: int = 101,
) -> list[CylPlaneEvaluation]:
    """批量评估 CylPlaneTerm 列表。"""
    out: list[CylPlaneEvaluation] = []
    for term in terms:
        out.append(
            evaluate_cyl_plane_term(
                cad_id=term.cad_id,
                scan_id=term.scan_id,
                o_cad=term.o_cad,
                v_cad=term.v_cad,
                n_scan=term.n_scan,
                q_scan=term.q_scan,
                target_axis_plane_dist_mm=term.target_axis_plane_dist_mm,
                R=R,
                t=t,
                axis_sample_half_length=axis_sample_half_length,
                axis_sample_count=axis_sample_count,
            )
        )
    return out


def evaluate_cyl_cyl_terms(terms: Iterable, scan_cyl_map: dict[int, object], R: np.ndarray, t: np.ndarray) -> list[CylCylEvaluation]:
    """
    批量评估 CylCylTerm 列表。

    说明
    ----
    cyl-cyl 评估里需要 Scan 圆柱 patch 上的样本点。
    当前 term 里未直接保存 scan_points，因此这里通过 scan_cyl_map[scan_id].mesh
    动态提取三角面中心。
    """
    out: list[CylCylEvaluation] = []
    for term in terms:
        scan_feat = scan_cyl_map[term.scan_id]
        scan_points = triangle_centers_from_mesh(scan_feat.mesh)

        out.append(
            evaluate_cyl_cyl_term(
                cad_id=term.cad_id,
                scan_id=term.scan_id,
                o_cad=term.o_cad,
                v_cad=term.v_cad,
                r_cad=term.r_cad,
                scan_points=scan_points,
                R=R,
                t=t,
            )
        )
    return out


# ============================================================
# 输出辅助
# ============================================================

def evaluation_to_dict(obj):
    """dataclass -> dict，方便日志保存或 JSON 序列化。"""
    return asdict(obj)


def format_distance_stats(name: str, stats: DistanceStats) -> str:
    """将平面/轴线到平面的距离统计格式化为易读文本。"""
    return (
        f"{name}\n"
        f"  count                = {stats.count}\n"
        f"  signed min / max     = {stats.signed_min:.6f} / {stats.signed_max:.6f}\n"
        f"  signed mean          = {stats.signed_mean:.6f}\n"
        f"  abs min / max        = {stats.abs_min:.6f} / {stats.abs_max:.6f}\n"
        f"  abs mean             = {stats.abs_mean:.6f}\n"
        f"  target d             = {stats.target_d:.6f}\n"
        f"  err(d) min / max     = {stats.err_to_target_min:.6f} / {stats.err_to_target_max:.6f}\n"
        f"  err(d) mean          = {stats.err_to_target_mean:.6f}\n"
        f"  err(d) abs mean      = {stats.err_to_target_abs_mean:.6f}"
    )


def format_cylinder_stats(name: str, stats: CylinderRadialStats) -> str:
    """将圆柱径向统计格式化为易读文本。"""
    return (
        f"{name}\n"
        f"  count                = {stats.count}\n"
        f"  radial min / max     = {stats.radial_min:.6f} / {stats.radial_max:.6f}\n"
        f"  radial mean          = {stats.radial_mean:.6f}\n"
        f"  target radius        = {stats.target_radius:.6f}\n"
        f"  err(r) min / max     = {stats.err_to_radius_min:.6f} / {stats.err_to_radius_max:.6f}\n"
        f"  err(r) mean          = {stats.err_to_radius_mean:.6f}\n"
        f"  err(r) abs mean      = {stats.err_to_radius_abs_mean:.6f}"
    )
