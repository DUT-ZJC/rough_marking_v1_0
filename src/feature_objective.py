from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ============================================================
# 基础工具
# ============================================================

def unit(v: np.ndarray) -> np.ndarray:
    """
    单位化向量。
    若长度过小，返回原向量对应的安全结果。
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def triangle_centers_from_mesh(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    计算三角网格中所有三角面的中心点。

    参数
    ----
    mesh : o3d.geometry.TriangleMesh

    返回
    ----
    centers : (N, 3) np.ndarray
        每个三角面的中心点。
    """
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


def downsample_points(points: np.ndarray, max_points: int | None = None) -> np.ndarray:
    """
    对点集做简单下采样。
    若点数不超过 max_points，则原样返回。

    这里使用“按索引均匀抽样”的简单策略，优点是稳定、实现简单。

    参数
    ----
    points : (N, 3) np.ndarray
    max_points : int | None

    返回
    ----
    sampled : (M, 3) np.ndarray
    """
    points = np.asarray(points, dtype=np.float64)
    if max_points is None or len(points) <= max_points:
        return points

    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int32)
    return points[idx]


def skew(w: np.ndarray) -> np.ndarray:
    """
    向量对应的反对称矩阵 [w]_x。
    """
    wx, wy, wz = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -wz,  wy],
            [wz,  0.0, -wx],
            [-wy, wx,  0.0],
        ],
        dtype=np.float64,
    )


# ============================================================
# 优化项数据结构
# ============================================================

@dataclass
class PlanePlaneTerm:
    """
    平面-平面约束项。

    说明
    ----
    - CAD 平面使用解析参数 (n_cad, q_cad)
    - Scan 平面使用：
        1) 解析法向 n_scan，用于方向一致性约束
        2) patch 上采样点 scan_points，用于“点到 CAD 解析平面”的距离约束
    """
    cad_id: int
    scan_id: int

    # CAD 解析平面：法向 + 平面上一点
    n_cad: np.ndarray            # (3,), unit
    q_cad: np.ndarray            # (3,)

    # Scan 解析平面法向（方向一致性项用）
    n_scan: np.ndarray           # (3,), unit

    # Scan 平面 patch 上的采样点（距离项用）
    scan_points: np.ndarray      # (N, 3)

    # 开关与容差
    enable_angle: bool
    angle_tol_deg: float

    enable_gap: bool
    target_gap_mm: float
    gap_tol_mm: float


@dataclass
class CylCylTerm:
    """
    圆柱-圆柱约束项。

    当前使用解析几何级约束：
    - 轴向一致
    - 轴线偏移（同轴偏移）
    """
    cad_id: int
    scan_id: int

    o_cad: np.ndarray            # (3,), CAD 轴上一点
    v_cad: np.ndarray            # (3,), unit CAD 轴方向
    r_cad: float

    o_scan: np.ndarray           # (3,), Scan 轴上一点
    v_scan: np.ndarray           # (3,), unit Scan 轴方向
    r_scan: float

    enable_axis_angle: bool
    axis_angle_tol_deg: float

    enable_axis_offset: bool
    axis_offset_tol_mm: float


@dataclass
class CylPlaneTerm:
    """
    圆柱-平面约束项。

    当前使用解析几何级约束：
    - CAD 圆柱轴线 与 Scan 平面法向 的夹角关系
    - CAD 圆柱轴线 到 Scan 平面 的法向距离
    """
    cad_id: int
    scan_id: int

    o_cad: np.ndarray            # (3,), CAD 轴上一点
    v_cad: np.ndarray            # (3,), unit CAD 轴方向

    n_scan: np.ndarray           # (3,), unit Scan 平面法向
    q_scan: np.ndarray           # (3,), Scan 平面上一点

    enable_axis_plane_angle: bool
    axis_plane_angle_tol_deg: float

    enable_axis_plane_dist: bool
    target_axis_plane_dist_mm: float
    axis_plane_dist_tol_mm: float


@dataclass
class RigidFeatureOptimizationProblem:
    """
    刚体特征优化问题。

    变量采用“相对粗配准初值的增量参数化”：
        x = [dwx, dwy, dwz, dtx, dty, dtz]

    其中：
    - dw* 是相对初值 R0 的小旋转向量
    - dt* 是相对初值 t0 的平移增量
    """
    R0: np.ndarray                      # (3, 3)
    t0: np.ndarray                      # (3,)
    rot_bound_deg: np.ndarray           # (3,), 每个旋转分量允许偏离初值的角度界

    plane_plane_terms: list[PlanePlaneTerm]
    cyl_cyl_terms: list[CylCylTerm]
    cyl_plane_terms: list[CylPlaneTerm]


# ============================================================
# 从主函数已有特征构建优化项
# ============================================================

def build_feature_terms(
    pair_constraints,
    cad_planes,
    cad_cyls,
    scan_planes,
    scan_cyls,
    max_plane_points: int | None = 1000,
):
    """
    将 GUI 中用户选出的 PairConstraintSpec 列表，
    翻译成优化器直接可用的 term 列表。

    参数
    ----
    pair_constraints : list[PairConstraintSpec]
        来自 picker.run() 的输出。

    cad_planes, cad_cyls, scan_planes, scan_cyls :
        主函数已有的原始特征对象列表。

    max_plane_points : int | None
        每个 Scan 平面 patch 最多保留多少个样本点（三角面中心）。
        若为 None，则不下采样。

    返回
    ----
    plane_plane_terms, cyl_cyl_terms, cyl_plane_terms
    """
    cad_plane_map = {p.id: p for p in cad_planes}
    cad_cyl_map = {c.id: c for c in cad_cyls}
    scan_plane_map = {p.id: p for p in scan_planes}
    scan_cyl_map = {c.id: c for c in scan_cyls}

    plane_plane_terms: list[PlanePlaneTerm] = []
    cyl_cyl_terms: list[CylCylTerm] = []
    cyl_plane_terms: list[CylPlaneTerm] = []

    for spec in pair_constraints:
        if spec.kind == "plane_plane":
            if spec.cad_id not in cad_plane_map or spec.scan_id not in scan_plane_map:
                continue

            cp = cad_plane_map[spec.cad_id]
            sp = scan_plane_map[spec.scan_id]

            # Scan 平面 patch 上所有三角面中心参与“点到 CAD 解析平面”的距离约束
            scan_points = triangle_centers_from_mesh(sp.mesh)
            scan_points = downsample_points(scan_points, max_plane_points)

            plane_plane_terms.append(
                PlanePlaneTerm(
                    cad_id=cp.id,
                    scan_id=sp.id,
                    n_cad=unit(np.asarray(cp.normal, dtype=np.float64)),
                    q_cad=np.asarray(cp.p0, dtype=np.float64),
                    n_scan=unit(np.asarray(sp.normal, dtype=np.float64)),
                    scan_points=scan_points,
                    enable_angle=bool(spec.enable_angle),
                    angle_tol_deg=float(spec.angle_tol_deg),
                    enable_gap=bool(spec.enable_gap),
                    target_gap_mm=float(spec.target_gap_mm),
                    gap_tol_mm=float(spec.gap_tol_mm),
                )
            )

        elif spec.kind == "cyl_cyl":
            if spec.cad_id not in cad_cyl_map or spec.scan_id not in scan_cyl_map:
                continue

            cc = cad_cyl_map[spec.cad_id]
            sc = scan_cyl_map[spec.scan_id]

            cyl_cyl_terms.append(
                CylCylTerm(
                    cad_id=cc.id,
                    scan_id=sc.id,
                    o_cad=np.asarray(cc.axis_origin, dtype=np.float64),
                    v_cad=unit(np.asarray(cc.axis_dir, dtype=np.float64)),
                    r_cad=float(cc.radius),
                    o_scan=np.asarray(sc.axis_origin, dtype=np.float64),
                    v_scan=unit(np.asarray(sc.axis_dir, dtype=np.float64)),
                    r_scan=float(sc.radius),
                    enable_axis_angle=bool(spec.enable_axis_angle),
                    axis_angle_tol_deg=float(spec.axis_angle_tol_deg),
                    enable_axis_offset=bool(spec.enable_axis_offset),
                    axis_offset_tol_mm=float(spec.axis_offset_tol_mm),
                )
            )

        elif spec.kind == "cyl_plane":
            if spec.cad_id not in cad_cyl_map or spec.scan_id not in scan_plane_map:
                continue

            cc = cad_cyl_map[spec.cad_id]
            sp = scan_plane_map[spec.scan_id]

            cyl_plane_terms.append(
                CylPlaneTerm(
                    cad_id=cc.id,
                    scan_id=sp.id,
                    o_cad=np.asarray(cc.axis_origin, dtype=np.float64),
                    v_cad=unit(np.asarray(cc.axis_dir, dtype=np.float64)),
                    n_scan=unit(np.asarray(sp.normal, dtype=np.float64)),
                    q_scan=np.asarray(sp.centroid, dtype=np.float64),
                    enable_axis_plane_angle=bool(spec.enable_axis_plane_angle),
                    axis_plane_angle_tol_deg=float(spec.axis_plane_angle_tol_deg),
                    enable_axis_plane_dist=bool(spec.enable_axis_plane_dist),
                    target_axis_plane_dist_mm=float(spec.target_axis_plane_dist_mm),
                    axis_plane_dist_tol_mm=float(spec.axis_plane_dist_tol_mm),
                )
            )

    return plane_plane_terms, cyl_cyl_terms, cyl_plane_terms


# ============================================================
# 刚体特征优化器
# ============================================================

class RigidFeatureOptimizer:
    """
    刚体特征优化器。

    设计目标
    --------
    - 不使用 ICP
    - 只使用解析特征与特征 patch 样本点构建残差
    - 旋转在粗配准初值附近有界
    - 平移不设边界
    """

    def __init__(self, problem: RigidFeatureOptimizationProblem):
        self.problem = problem

    # ----------------------------
    # 参数化与边界
    # ----------------------------

    def pack_initial(self) -> np.ndarray:
        """
        返回相对初值的优化变量初值：
            x0 = [0, 0, 0, 0, 0, 0]
        """
        return np.zeros(6, dtype=np.float64)

    def bounds(self):
        """
        旋转增量有界，平移无界。
        """
        b = np.deg2rad(np.asarray(self.problem.rot_bound_deg, dtype=np.float64).reshape(3))
        lb = np.array([-b[0], -b[1], -b[2], -np.inf, -np.inf, -np.inf], dtype=np.float64)
        ub = np.array([ b[0],  b[1],  b[2],  np.inf,  np.inf,  np.inf], dtype=np.float64)
        return lb, ub

    def unpack_transform(self, x: np.ndarray):
        """
        将优化变量 x 解包成刚体变换 (R, t)。

        参数
        ----
        x : (6,)
            [dwx, dwy, dwz, dtx, dty, dtz]

        返回
        ----
        R : (3,3)
        t : (3,)
        """
        x = np.asarray(x, dtype=np.float64).reshape(6)
        dtheta = x[:3]
        dt = x[3:]

        # 使用旋转向量参数化相对初值的小旋转
        dR = Rotation.from_rotvec(dtheta).as_matrix()

        # 最终旋转和平移
        R = self.problem.R0 @ dR
        t = self.problem.t0 + dt
        return R, t

    # ----------------------------
    # 权重
    # ----------------------------

    @staticmethod
    def _weight_from_tol(tol: float, eps: float = 1e-12) -> float:
        """
        根据容差生成权重：
            w = 1 / tol^2

        这样容差越小，约束越强。
        """
        tol = float(tol)
        return 1.0 / max(tol * tol, eps)

    # ----------------------------
    # 各类残差
    # ----------------------------

    def _append_plane_plane_residuals(self, residuals: list[float], R: np.ndarray, t: np.ndarray):
        """
        追加所有 plane-plane 约束残差。

        数学形式
        --------
        1) 方向一致项（双向不敏感）：
            r_ang = 1 - (n_s^T (R n_c))^2

        2) 多点距离项：
            对 scan patch 上每个样本点 p_i，
            r_i = (R n_c)^T [ p_i - (R q_c + t) ] - d*
        """
        for term in self.problem.plane_plane_terms:
            n_cad_p = R @ term.n_cad
            q_cad_p = R @ term.q_cad + t

            # 方向一致项：正反向都认为一致，所以使用平方型
            if term.enable_angle:
                sigma_ang = np.deg2rad(term.angle_tol_deg)
                w_ang = self._weight_from_tol(sigma_ang)

                r_ang = 1.0 - (np.dot(term.n_scan, n_cad_p) ** 2)
                residuals.append(np.sqrt(w_ang) * r_ang)

            # 多点距离项：scan plane patch 上所有采样点都参与
            if term.enable_gap and len(term.scan_points) > 0:
                # 除以样本数，避免大平面因为点多而天然权重更大
                w_gap = self._weight_from_tol(term.gap_tol_mm) / max(len(term.scan_points), 1)

                for p in term.scan_points:
                    r_gap = np.dot(n_cad_p, p - q_cad_p) - term.target_gap_mm
                    residuals.append(np.sqrt(w_gap) * r_gap)

    def _append_cyl_cyl_residuals(self, residuals: list[float], R: np.ndarray, t: np.ndarray):
        """
        追加所有 cyl-cyl 约束残差。

        数学形式
        --------
        1) 轴向一致项（双向不敏感）：
            r_ang = 1 - (v_s^T (R v_c))^2

        2) 轴线偏移项：
            r_off = || (I - v_s v_s^T) ( (R o_c + t) - o_s ) ||
        """
        I = np.eye(3, dtype=np.float64)

        for term in self.problem.cyl_cyl_terms:
            o_cad_p = R @ term.o_cad + t
            v_cad_p = R @ term.v_cad

            if term.enable_axis_angle:
                sigma_ang = np.deg2rad(term.axis_angle_tol_deg)
                w_ang = self._weight_from_tol(sigma_ang)

                r_ang = 1.0 - (np.dot(term.v_scan, v_cad_p) ** 2)
                residuals.append(np.sqrt(w_ang) * r_ang)

            if term.enable_axis_offset:
                w_off = self._weight_from_tol(term.axis_offset_tol_mm)

                proj = I - np.outer(term.v_scan, term.v_scan)
                r_off = np.linalg.norm(proj @ (o_cad_p - term.o_scan))
                residuals.append(np.sqrt(w_off) * r_off)

    def _append_cyl_plane_residuals(self, residuals: list[float], R: np.ndarray, t: np.ndarray):
        """
        追加所有 cyl-plane 约束残差。

        数学形式
        --------
        1) 轴线与平面法向关系：
            若轴线应平行于平面，则轴线应垂直于法向
            r_ang = (n_s^T (R v_c))^2

        2) 轴线上一点到平面的有符号距离：
            r_dist = n_s^T [ (R o_c + t) - q_s ] - d*
        """
        for term in self.problem.cyl_plane_terms:
            o_cad_p = R @ term.o_cad + t
            v_cad_p = R @ term.v_cad

            if term.enable_axis_plane_angle:
                sigma_ang = np.deg2rad(term.axis_plane_angle_tol_deg)
                w_ang = self._weight_from_tol(sigma_ang)

                r_ang = (np.dot(term.n_scan, v_cad_p) ** 2)
                residuals.append(np.sqrt(w_ang) * r_ang)

            if term.enable_axis_plane_dist:
                w_dist = self._weight_from_tol(term.axis_plane_dist_tol_mm)

                r_dist = np.dot(term.n_scan, o_cad_p - term.q_scan) - term.target_axis_plane_dist_mm
                residuals.append(np.sqrt(w_dist) * r_dist)

    # ----------------------------
    # 总残差向量
    # ----------------------------

    def residual_vector(self, x: np.ndarray) -> np.ndarray:
        """
        将所有约束项拼成一个总残差向量，供 least_squares 使用。
        """
        R, t = self.unpack_transform(x)
        residuals: list[float] = []

        self._append_plane_plane_residuals(residuals, R, t)
        self._append_cyl_cyl_residuals(residuals, R, t)
        self._append_cyl_plane_residuals(residuals, R, t)

        return np.asarray(residuals, dtype=np.float64)

    # ----------------------------
    # 求解
    # ----------------------------

    def solve(self):
        """
        调用 scipy.optimize.least_squares 求解。

        返回
        ----
        dict
            包含最终 R, t, T 以及求解状态。
        """
        x0 = self.pack_initial()
        lb, ub = self.bounds()

        result = least_squares(
            self.residual_vector,
            x0,
            bounds=(lb, ub),
            method="trf",
            verbose=2,
        )

        R_opt, t_opt = self.unpack_transform(result.x)

        T_opt = np.eye(4, dtype=np.float64)
        T_opt[:3, :3] = R_opt
        T_opt[:3, 3] = t_opt

        return {
            "ok": bool(result.success),
            "message": result.message,
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "x": result.x,
            "R": R_opt,
            "t": t_opt,
            "T": T_opt,
        }
