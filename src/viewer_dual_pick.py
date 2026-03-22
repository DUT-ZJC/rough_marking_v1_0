from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import platform


# =========================
# 数据结构定义
# =========================

@dataclass
class PairConstraintSpec:
    """描述一对特征之间的约束配置。"""

    kind: str  # "plane_plane" | "cyl_cyl" | "cyl_plane"
    cad_id: int
    scan_id: int

    # ---------- plane-plane ----------
    enable_angle: bool = True
    angle_tol_deg: float = 0.10

    enable_gap: bool = True
    target_gap_mm: float = 0.0
    gap_tol_mm: float = 0.10
    gap_ref: str = "cad_normal"  # 间距符号方向沿 CAD 平面法向

    # ---------- cyl-cyl ----------
    enable_axis_angle: bool = True
    axis_angle_tol_deg: float = 0.10

    enable_axis_offset: bool = True
    axis_offset_tol_mm: float = 0.10

    # ---------- cyl-plane ----------
    enable_axis_plane_angle: bool = True
    axis_plane_angle_tol_deg: float = 0.10

    enable_axis_plane_dist: bool = True
    target_axis_plane_dist_mm: float = 0.0
    axis_plane_dist_tol_mm: float = 0.10
    axis_plane_dist_ref: str = "plane_normal"


@dataclass
class FeatureInfo:
    """描述一个几何特征（平面或圆柱）的基础信息。"""

    side: str                 # "cad" | "scan"
    kind: str                 # "plane" | "cyl"
    fid: int
    mesh: o3d.geometry.TriangleMesh
    center: np.ndarray
    direction: np.ndarray     # 平面法向 or 圆柱轴向
    triangle_centers: np.ndarray
    avg_radius: float


@dataclass
class SelectionState:
    """描述当前被用户选中的特征状态。"""

    kind: str
    fid: int
    center: np.ndarray
    direction: np.ndarray
    feature: FeatureInfo


# =========================
# 几何工具函数
# =========================

def _is_left_click(event) -> bool:
    """判断事件是否为鼠标左键点击。"""
    try:
        if hasattr(event, "button") and event.button == gui.MouseButton.LEFT:
            return True
    except Exception:
        pass

    try:
        if hasattr(event, "is_button_down"):
            return bool(event.is_button_down(gui.MouseButton.LEFT))
    except Exception:
        pass

    try:
        if hasattr(event, "buttons"):
            return bool(event.buttons & int(gui.MouseButton.LEFT))
    except Exception:
        pass

    return False

def _setup_open3d_font() -> None:
    """
    为 Open3D GUI 配置中文字体。

    说明：
    1. 必须在 gui.Application.instance.initialize() 之后调用；
    2. 必须在 create_window() 之前调用；
    3. zh_all 能覆盖全部中文字符，但会占用更多内存。
    """
    system = platform.system()

    if system == "Darwin":
        hanzi = "STHeiti Light"
    elif system == "Windows":
        # Windows 下官方示例建议直接给字体文件路径
        hanzi = r"c:/windows/fonts/msyh.ttc"
    else:
        # Linux 下常见写法；要求系统已安装 Noto CJK
        hanzi = "NotoSansCJK"

    font = gui.FontDescription()
    font.add_typeface_for_language(hanzi, "zh_all")
    gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)


def _unit(v: np.ndarray) -> np.ndarray:
    """将向量单位化；若长度过小，则返回默认 z 轴方向。"""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / norm


def _rot_z_to_v(v: np.ndarray) -> np.ndarray:
    """返回把 +Z 方向旋转到向量 v 的旋转矩阵。"""
    v = _unit(v)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(np.dot(z, v))

    if abs(c - 1.0) < 1e-12:
        return np.eye(3)

    if abs(c + 1.0) < 1e-12:
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    axis = axis / (s + 1e-12)

    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )

    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _mesh_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """将 Open3D 顶点数组转换为 numpy 数组。"""
    return np.asarray(mesh.vertices, dtype=np.float64)


def _mesh_triangles(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """将 Open3D 三角形索引转换为 numpy 数组。"""
    return np.asarray(mesh.triangles, dtype=np.int32)


def _triangle_centers(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """计算网格中所有三角形面片的中心点。"""
    vertices = _mesh_vertices(mesh)
    triangles = _mesh_triangles(mesh)

    if len(vertices) == 0 or len(triangles) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    return (
        vertices[triangles[:, 0]]
        + vertices[triangles[:, 1]]
        + vertices[triangles[:, 2]]
    ) / 3.0


def _estimate_plane_normal(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """通过三角面法向平均估计平面法向。"""
    vertices = _mesh_vertices(mesh)
    triangles = _mesh_triangles(mesh)

    if len(vertices) == 0 or len(triangles) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )

    normal_norm = np.linalg.norm(normals, axis=1)
    valid = normal_norm > 1e-12
    if not np.any(valid):
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    normals = (normals[valid].T / normal_norm[valid]).T
    mean_normal = np.mean(normals, axis=0)
    return _unit(mean_normal)


def _estimate_cylinder_axis(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """通过 PCA 估计圆柱特征的主轴方向。"""
    vertices = _mesh_vertices(mesh)
    if len(vertices) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    center = np.mean(vertices, axis=0)
    X = vertices - center
    cov = X.T @ X

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        axis = eigenvectors[:, np.argmax(eigenvalues)]
        return _unit(axis)
    except Exception:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _estimate_avg_radius(mesh: o3d.geometry.TriangleMesh, center: np.ndarray) -> float:
    """
    估计一个特征的大致尺寸，用于绘制箭头。
    这里不是严格几何意义上的圆柱半径，只是用于可视化缩放。
    """
    vertices = _mesh_vertices(mesh)
    if len(vertices) == 0:
        return 1.0

    distances = np.linalg.norm(vertices - center[None, :], axis=1)
    return float(max(np.mean(distances), 1e-3))



# =========================
# 主程序
# =========================

class DualPickerApp:
    """CAD 与扫描件双视图特征选择界面。"""

    def __init__(
        self,
        cad_plane_features,
        cad_cyl_features,
        scan_plane_features,
        scan_cyl_features,
        cad_base_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        scan_base_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        width: int = 1600,
        height: int = 900,
    )-> None:
        # ---------- 原始数据 ----------
        self.cad_base_mesh = cad_base_mesh
        self.scan_base_mesh = scan_base_mesh

        self.cad_plane_triangle_map = self._build_triangle_map(cad_plane_features)
        self.cad_cyl_triangle_map = self._build_triangle_map(cad_cyl_features)
        self.scan_plane_triangle_map = self._build_triangle_map(scan_plane_features)
        self.scan_cyl_triangle_map = self._build_triangle_map(scan_cyl_features)


        # ---------- 当前选择与约束 ----------
        self.current_cad: Optional[SelectionState] = None
        self.current_scan: Optional[SelectionState] = None
        self.constraints: list[PairConstraintSpec] = []

        # ---------- 预处理特征 ----------
        self.cad_plane_feats = self._build_feature_infos_from_objects("cad", "plane", cad_plane_features)
        self.cad_cyl_feats = self._build_feature_infos_from_objects("cad", "cyl", cad_cyl_features)
        self.scan_plane_feats = self._build_feature_infos_from_objects("scan", "plane", scan_plane_features)
        self.scan_cyl_feats = self._build_feature_infos_from_objects("scan", "cyl", scan_cyl_features)


        self.cad_all_feats: list[FeatureInfo] = (
            list(self.cad_plane_feats.values()) + list(self.cad_cyl_feats.values())
        )
        self.scan_all_feats: list[FeatureInfo] = (
            list(self.scan_plane_feats.values()) + list(self.scan_cyl_feats.values())
        )

        # ---------- 布局参数 ----------
        self._panel_w = 420
        self._gap = 10

        # ---------- GUI 初始化 ----------
        app = gui.Application.instance
        app.initialize()

        _setup_open3d_font()

        self.app = app


        self.win = app.create_window("CAD / 扫描件 特征约束选择", width, height)
        self.win.set_on_layout(self._on_layout)

        self.cad_scene = gui.SceneWidget()
        self.scan_scene = gui.SceneWidget()
        self.cad_scene.scene = rendering.Open3DScene(self.win.renderer)
        self.scan_scene.scene = rendering.Open3DScene(self.win.renderer)

        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.panel.preferred_width = self._panel_w

        self.win.add_child(self.cad_scene)
        self.win.add_child(self.scan_scene)
        self.win.add_child(self.panel)

        # ---------- 射线求交 ----------
        self.ray_cad_base = o3d.t.geometry.RaycastingScene()
        self.ray_scan_base = o3d.t.geometry.RaycastingScene()
        self._cad_base_gid = None
        self._scan_base_gid = None

        # ---------- 高亮名称 ----------
        self._cad_hl_name = "__cad_highlight__"
        self._scan_hl_name = "__scan_highlight__"
        self._cad_arrow_name = "__cad_arrow__"
        self._scan_arrow_name = "__scan_arrow__"

        self._cad_hl_added = False
        self._scan_hl_added = False
        self._cad_arrow_added = False
        self._scan_arrow_added = False

        # ---------- 构建界面 ----------
        self._build_panel()
        self._add_geometries()

        gui.Application.instance.post_to_main_thread(self.win, self._setup_camera)

        # ---------- 鼠标事件 ----------
        self.cad_scene.set_on_mouse(self._on_mouse_cad)
        self.scan_scene.set_on_mouse(self._on_mouse_scan)

    # =========================
    # 特征预处理
    # =========================

    def _feature_color(self, kind: str, fid: int, side: str) -> tuple[float, float, float]:
        """
        为同类不同特征生成略有差异的颜色，便于区分边界。
        """
        if side == "cad":
            if kind == "plane":
                base = np.array([0.15, 0.75, 0.20], dtype=np.float64)
            else:
                base = np.array([0.15, 0.35, 0.95], dtype=np.float64)
        else:
            if kind == "plane":
                base = np.array([0.95, 0.85, 0.10], dtype=np.float64)
            else:
                base = np.array([0.90, 0.20, 0.85], dtype=np.float64)

        # 用 fid 做一点明暗扰动
        delta = ((fid % 7) - 3) * 0.04
        color = np.clip(base + delta, 0.0, 1.0)
        return float(color[0]), float(color[1]), float(color[2])

    def _mat_line(self, rgb, line_width: float = 2.0):
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.base_color = (rgb[0], rgb[1], rgb[2], 1.0)
        mat.line_width = line_width
        return mat

    def _mesh_edges_as_lineset(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet | None:
        """把三角网格转成边线，用于增强边界显示。"""
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return None
        try:
            line = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            return line
        except Exception:
            return None

    def _build_feature_infos_from_objects(self, side: str, kind: str, features) -> dict[int, FeatureInfo]:
        out: dict[int, FeatureInfo] = {}

        for feat in features:
            mesh = copy.deepcopy(feat.mesh)
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                continue

            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            fid = int(feat.id)

            if kind == "plane":
                direction = _unit(np.asarray(feat.normal, dtype=np.float64))

                if hasattr(feat, "p0"):
                    center = np.asarray(feat.p0, dtype=np.float64)
                elif hasattr(feat, "centroid"):
                    center = np.asarray(feat.centroid, dtype=np.float64)
                else:
                    center = np.asarray(mesh.get_center(), dtype=np.float64)

                avg_radius = _estimate_avg_radius(mesh, center)

            else:
                direction = _unit(np.asarray(feat.axis_dir, dtype=np.float64))

                if hasattr(feat, "axis_origin"):
                    center = np.asarray(feat.axis_origin, dtype=np.float64)
                else:
                    center = np.asarray(mesh.get_center(), dtype=np.float64)

                avg_radius = float(getattr(feat, "radius", _estimate_avg_radius(mesh, center)))

            out[fid] = FeatureInfo(
                side=side,
                kind=kind,
                fid=fid,
                mesh=mesh,
                center=center,
                direction=direction,
                triangle_centers=_triangle_centers(mesh),
                avg_radius=avg_radius,
            )

        return out


    def _build_triangle_map(self, features) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for feat in features:
            tri_ids = getattr(feat, "tri_indices", None)
            if tri_ids is None:
                continue
            arr = np.asarray(tri_ids, dtype=np.int32).reshape(-1)
            if arr.size == 0:
                continue
            out[int(feat.id)] = arr
        return out
    #创建特征到mesh的特征索引映射
    
    def _get_feature_mesh_from_base(self, side: str, kind: str, fid: int) -> o3d.geometry.TriangleMesh | None:
        base_mesh = self.cad_base_mesh if side == "cad" else self.scan_base_mesh
        if base_mesh is None:
            return None
    
        tri_ids = self._feature_triangles(side, kind, fid)
        if tri_ids.size == 0:
            return None
    
        mesh = self._extract_submesh_by_triangles(base_mesh, tri_ids)
        if mesh is not None and not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh
    #获取feature_mesh


    # =========================
    # 布局
    # =========================

    def _on_layout(self, ctx) -> None:
        """窗口布局回调：左右两个视图，中间偏右是控制面板。"""
        rect = self.win.content_rect
        gap = self._gap
        panel_w = self._panel_w

        usable_w = max(1, rect.width - panel_w - 3 * gap)
        scene_w = max(1, usable_w // 2)
        x0 = rect.x + gap
        y0 = rect.y + gap
        h = max(1, rect.height - 2 * gap)

        self.cad_scene.frame = gui.Rect(x0, y0, scene_w, h)
        self.scan_scene.frame = gui.Rect(x0 + scene_w + gap, y0, scene_w, h)

        panel_x = x0 + 2 * scene_w + 2 * gap
        panel_real_w = max(1, rect.width - (panel_x - rect.x) - gap)
        self.panel.frame = gui.Rect(panel_x, y0, panel_real_w, h)

    # =========================
    # 材质
    # =========================

    def _mat_lit(self, rgb, alpha: float = 1.0) -> rendering.MaterialRecord:
        """创建受光照影响的材质。"""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = (rgb[0], rgb[1], rgb[2], alpha)
        return mat

    def _mat_unlit(self, rgb, alpha: float = 1.0) -> rendering.MaterialRecord:
        """创建不受光照影响的材质。"""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = (rgb[0], rgb[1], rgb[2], alpha)
        return mat

    # =========================
    # 网格拆分与场景添加
    # =========================

    def _extract_submesh_by_triangles(
        self,
        mesh: o3d.geometry.TriangleMesh,
        tri_ids,
    ) -> o3d.geometry.TriangleMesh | None:
        """根据三角形索引，从原始网格中提取子网格。"""
        tri_ids = np.asarray(tri_ids, dtype=np.int32).reshape(-1)
        if mesh is None or len(tri_ids) == 0:
            return None

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        tri_ids = tri_ids[(tri_ids >= 0) & (tri_ids < len(triangles))]
        if len(tri_ids) == 0:
            return None

        sub_triangles = triangles[tri_ids]
        used_vertex_ids = np.unique(sub_triangles.reshape(-1))

        remap = -np.ones(len(vertices), dtype=np.int32)
        remap[used_vertex_ids] = np.arange(len(used_vertex_ids), dtype=np.int32)

        out = o3d.geometry.TriangleMesh()
        out.vertices = o3d.utility.Vector3dVector(vertices[used_vertex_ids])
        out.triangles = o3d.utility.Vector3iVector(remap[sub_triangles])

        if mesh.has_vertex_normals():
            vertex_normals = np.asarray(mesh.vertex_normals)
            if len(vertex_normals) == len(vertices):
                out.vertex_normals = o3d.utility.Vector3dVector(
                    vertex_normals[used_vertex_ids]
                )

        if not out.has_vertex_normals():
            out.compute_vertex_normals()

        return out

    def _build_boundary_lineset(
        self,
        mesh: o3d.geometry.TriangleMesh,
        plane_map: dict[int, np.ndarray],
        cyl_map: dict[int, np.ndarray],
    ) -> o3d.geometry.LineSet | None:
        """
        只提取“标签发生变化”的边界线：
        - 非特征 <-> 特征
        - 特征 <-> 特征
        不显示特征内部三角剖分线。
        """
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return None

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        n_tri = len(triangles)

        # 1) 给每个 triangle 打标签
        #    0 表示背景
        #    平面用正数：100000 + fid
        #    圆柱用负数：-100000 - fid
        tri_label = np.zeros(n_tri, dtype=np.int64)

        for fid, tri_ids in plane_map.items():
            arr = np.asarray(tri_ids, dtype=np.int32).reshape(-1)
            arr = arr[(arr >= 0) & (arr < n_tri)]
            tri_label[arr] = 100000 + int(fid)

        for fid, tri_ids in cyl_map.items():
            arr = np.asarray(tri_ids, dtype=np.int32).reshape(-1)
            arr = arr[(arr >= 0) & (arr < n_tri)]
            tri_label[arr] = -100000 - int(fid)

        # 2) 建立 edge -> 相邻 triangle 列表
        edge_to_tris: dict[tuple[int, int], list[int]] = {}

        for tid, tri in enumerate(triangles):
            e0 = tuple(sorted((int(tri[0]), int(tri[1]))))
            e1 = tuple(sorted((int(tri[1]), int(tri[2]))))
            e2 = tuple(sorted((int(tri[2]), int(tri[0]))))

            edge_to_tris.setdefault(e0, []).append(tid)
            edge_to_tris.setdefault(e1, []).append(tid)
            edge_to_tris.setdefault(e2, []).append(tid)

        # 3) 只保留“边两侧 triangle 标签不同”的 edge
        boundary_edges = []

        for edge, tri_list in edge_to_tris.items():
            if len(tri_list) == 1:
                # 裸边：如果你希望模型外轮廓也画出来，可以保留
                # 这里只在它属于特征时保留；纯背景裸边可不画
                t0 = tri_list[0]
                if tri_label[t0] != 0:
                    boundary_edges.append(edge)
                continue

            if len(tri_list) >= 2:
                t0, t1 = tri_list[0], tri_list[1]
                if tri_label[t0] != tri_label[t1]:
                    boundary_edges.append(edge)

        if len(boundary_edges) == 0:
            return None

        lines = np.asarray(boundary_edges, dtype=np.int32)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        return line_set


    def _split_mesh_by_feature_maps(
        self,
        mesh: o3d.geometry.TriangleMesh,
        plane_map: dict[int, np.ndarray],
        cyl_map: dict[int, np.ndarray],
    ) -> dict[str, o3d.geometry.TriangleMesh | None]:
        """
        按特征三角形映射将 base mesh 分成三部分：
        - bg: 非特征背景
        - plane: 所有平面特征区域
        - cyl: 所有圆柱特征区域
        """
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        num_triangles = len(triangles)

        plane_ids = []
        for arr in plane_map.values():
            plane_ids.append(np.asarray(arr, dtype=np.int32).reshape(-1))
        plane_ids = (
            np.concatenate(plane_ids)
            if len(plane_ids) > 0
            else np.zeros((0,), dtype=np.int32)
        )

        cyl_ids = []
        for arr in cyl_map.values():
            cyl_ids.append(np.asarray(arr, dtype=np.int32).reshape(-1))
        cyl_ids = (
            np.concatenate(cyl_ids)
            if len(cyl_ids) > 0
            else np.zeros((0,), dtype=np.int32)
        )

        plane_ids = plane_ids[(plane_ids >= 0) & (plane_ids < num_triangles)]
        cyl_ids = cyl_ids[(cyl_ids >= 0) & (cyl_ids < num_triangles)]

        feat_mask = np.zeros(num_triangles, dtype=bool)
        feat_mask[plane_ids] = True
        feat_mask[cyl_ids] = True

        bg_ids = np.nonzero(~feat_mask)[0]

        return {
            "bg": self._extract_submesh_by_triangles(mesh, bg_ids),
            "plane": self._extract_submesh_by_triangles(mesh, plane_ids),
            "cyl": self._extract_submesh_by_triangles(mesh, cyl_ids),
        }

    def _feature_triangles(self, side: str, kind: str, fid: int) -> np.ndarray:
        """根据 side / kind / fid 返回对应特征的三角形索引。"""
        if side == "cad":
            feat_map = (
                self.cad_plane_triangle_map if kind == "plane" else self.cad_cyl_triangle_map
            )
        else:
            feat_map = (
                self.scan_plane_triangle_map if kind == "plane" else self.scan_cyl_triangle_map
            )
        return np.asarray(feat_map.get(fid, []), dtype=np.int32)

    def _add_geometries(self):
        print(">>> ENTER _add_geometries <<<")
    
        self.cad_scene.scene.set_background([1, 1, 1, 1])
        self.scan_scene.scene.set_background([1, 1, 1, 1])
    
        bg_mat = self._mat_lit((0.78, 0.78, 0.80), 1.0)
        cad_plane_mat = self._mat_lit((0.22, 0.78, 0.28), 1.0)
        cad_cyl_mat = self._mat_lit((0.20, 0.38, 0.95), 1.0)
        scan_plane_mat = self._mat_lit((0.95, 0.85, 0.15), 1.0)
        scan_cyl_mat = self._mat_lit((0.88, 0.25, 0.82), 1.0)
        boundary_mat = self._mat_line((0.05, 0.05, 0.05), 1.0)
    
        # ---------- CAD ----------
        if self.cad_base_mesh is not None and len(self.cad_base_mesh.triangles) > 0:
            cad_mesh = copy.deepcopy(self.cad_base_mesh)
            if not cad_mesh.has_vertex_normals():
                cad_mesh.compute_vertex_normals()
    
            cad_parts = self._split_mesh_by_feature_maps(
                cad_mesh,
                self.cad_plane_triangle_map,
                self.cad_cyl_triangle_map,
            )
    
            print("cad plane is None:", cad_parts["plane"] is None)
            print("cad cyl is None:", cad_parts["cyl"] is None)
    
            # 背景
            if cad_parts["bg"] is not None:
                self.cad_scene.scene.add_geometry("cad_bg", cad_parts["bg"], bg_mat)
    
            # 所有 plane 合并区域
            if cad_parts["plane"] is not None:
                print(">>> add cad plane <<<", len(cad_parts["plane"].triangles))
                self.cad_scene.scene.add_geometry("cad_plane_cls", cad_parts["plane"], cad_plane_mat)
    
            # 所有 cyl 合并区域
            if cad_parts["cyl"] is not None:
                print(">>> add cad cyl <<<", len(cad_parts["cyl"].triangles))
                self.cad_scene.scene.add_geometry("cad_cyl_cls", cad_parts["cyl"], cad_cyl_mat)
    
            # 只叠加“标签变化边界”
            cad_boundary = self._build_boundary_lineset(
                cad_mesh,
                self.cad_plane_triangle_map,
                self.cad_cyl_triangle_map,
            )
            if cad_boundary is not None:
                self.cad_scene.scene.add_geometry("cad_boundaries", cad_boundary, boundary_mat)
    
            # 射线拾取场景
            self._cad_base_gid = int(
                self.ray_cad_base.add_triangles(
                    o3d.t.geometry.TriangleMesh.from_legacy(cad_mesh)
                )
            )
            print(">>> CAD raycasting triangles <<<", len(cad_mesh.triangles))
    
        # ---------- SCAN ----------
        if self.scan_base_mesh is not None and len(self.scan_base_mesh.triangles) > 0:
            scan_mesh = copy.deepcopy(self.scan_base_mesh)
            if not scan_mesh.has_vertex_normals():
                scan_mesh.compute_vertex_normals()
    
            scan_parts = self._split_mesh_by_feature_maps(
                scan_mesh,
                self.scan_plane_triangle_map,
                self.scan_cyl_triangle_map,
            )
    
            print("scan plane is None:", scan_parts["plane"] is None)
            print("scan cyl is None:", scan_parts["cyl"] is None)
    
            # 背景
            if scan_parts["bg"] is not None:
                self.scan_scene.scene.add_geometry("scan_bg", scan_parts["bg"], bg_mat)
    
            # 所有 plane 合并区域
            if scan_parts["plane"] is not None:
                print(">>> add scan plane <<<", len(scan_parts["plane"].triangles))
                self.scan_scene.scene.add_geometry("scan_plane_cls", scan_parts["plane"], scan_plane_mat)
    
            # 所有 cyl 合并区域
            if scan_parts["cyl"] is not None:
                print(">>> add scan cyl <<<", len(scan_parts["cyl"].triangles))
                self.scan_scene.scene.add_geometry("scan_cyl_cls", scan_parts["cyl"], scan_cyl_mat)
    
            # 只叠加“标签变化边界”
            scan_boundary = self._build_boundary_lineset(
                scan_mesh,
                self.scan_plane_triangle_map,
                self.scan_cyl_triangle_map,
            )
            if scan_boundary is not None:
                self.scan_scene.scene.add_geometry("scan_boundaries", scan_boundary, boundary_mat)
    
            # 射线拾取场景
            self._scan_base_gid = int(
                self.ray_scan_base.add_triangles(
                    o3d.t.geometry.TriangleMesh.from_legacy(scan_mesh)
                )
            )
            print(">>> SCAN raycasting triangles <<<", len(scan_mesh.triangles))




    def _setup_camera(self) -> None:
        """根据模型与特征包围盒初始化相机。"""

        def bbox_from(mesh, feats) -> o3d.geometry.AxisAlignedBoundingBox:
            bbox = o3d.geometry.AxisAlignedBoundingBox()
            has_any = False

            if mesh is not None and len(mesh.vertices) > 0:
                bbox += mesh.get_axis_aligned_bounding_box()
                has_any = True

            for feat in feats:
                bbox += feat.mesh.get_axis_aligned_bounding_box()
                has_any = True

            if not has_any:
                bbox.min_bound = np.array([-100.0, -100.0, -100.0], dtype=np.float64)
                bbox.max_bound = np.array([100.0, 100.0, 100.0], dtype=np.float64)

            return bbox

        bbox_cad = bbox_from(self.cad_base_mesh, self.cad_all_feats)
        bbox_scan = bbox_from(self.scan_base_mesh, self.scan_all_feats)

        self.cad_scene.setup_camera(60.0, bbox_cad, bbox_cad.get_center())
        self.scan_scene.setup_camera(60.0, bbox_scan, bbox_scan.get_center())

    # =========================
    # 控制面板
    # =========================

    def _build_panel(self) -> None:
        """构建右侧控制面板。"""
        title = gui.Label("约束控制")
        self.panel.add_child(title)
        self.panel.add_fixed(6)

        self.lbl_cad = gui.Label("当前 CAD: (none)")
        self.lbl_scan = gui.Label("当前 扫描件: (none)")
        self.lbl_hint = gui.Label("说明: 先点左侧 CAD，再点中间扫描件。选中后会显示法向/轴线箭头。")

        self.panel.add_child(self.lbl_cad)
        self.panel.add_child(self.lbl_scan)
        self.panel.add_child(self.lbl_hint)

        self.panel.add_fixed(8)
        self.panel.add_child(gui.Label("约束类型"))

        self.kind_combo = gui.Combobox()
        self.kind_combo.add_item("plane-plane")
        self.kind_combo.add_item("cyl-cyl")
        self.kind_combo.add_item("cyl-plane")
        self.kind_combo.selected_index = 0
        self.kind_combo.set_on_selection_changed(self._on_kind_changed)
        self.panel.add_child(self.kind_combo)

        # ---------- plane-plane ----------
        self.panel.add_fixed(8)
        self.panel.add_child(gui.Label("Plane-Plane"))

        self.pl_enable_angle = gui.Checkbox("角度约束")
        self.pl_enable_angle.checked = True

        self.pl_angle_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.pl_angle_tol.set_limits(0.001, 30.0)
        self.pl_angle_tol.double_value = 0.10

        self.pl_enable_gap = gui.Checkbox("间距约束 d (沿 CAD 法向，有符号)")
        self.pl_enable_gap.checked = True

        self.pl_target_gap = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.pl_target_gap.set_limits(-10000.0, 10000.0)
        self.pl_target_gap.double_value = 0.0

        self.pl_gap_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.pl_gap_tol.set_limits(0.001, 1000.0)
        self.pl_gap_tol.double_value = 0.10

        self.panel.add_child(self.pl_enable_angle)
        self.panel.add_child(gui.Label("平面夹角容差 (deg)"))
        self.panel.add_child(self.pl_angle_tol)
        self.panel.add_child(self.pl_enable_gap)
        self.panel.add_child(gui.Label("目标间距 d (mm)"))
        self.panel.add_child(self.pl_target_gap)
        self.panel.add_child(gui.Label("间距容差 (mm)"))
        self.panel.add_child(self.pl_gap_tol)

        # ---------- cyl-cyl ----------
        self.panel.add_fixed(8)
        self.panel.add_child(gui.Label("Cyl-Cyl"))

        self.cy_enable_axis_angle = gui.Checkbox("轴线夹角约束")
        self.cy_enable_axis_angle.checked = True

        self.cy_axis_angle_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cy_axis_angle_tol.set_limits(0.001, 30.0)
        self.cy_axis_angle_tol.double_value = 0.10

        self.cy_enable_axis_offset = gui.Checkbox("同轴偏移约束")
        self.cy_enable_axis_offset.checked = True

        self.cy_axis_offset_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cy_axis_offset_tol.set_limits(0.001, 1000.0)
        self.cy_axis_offset_tol.double_value = 0.10

        self.panel.add_child(self.cy_enable_axis_angle)
        self.panel.add_child(gui.Label("轴线夹角容差 (deg)"))
        self.panel.add_child(self.cy_axis_angle_tol)
        self.panel.add_child(self.cy_enable_axis_offset)
        self.panel.add_child(gui.Label("轴线偏移容差 (mm)"))
        self.panel.add_child(self.cy_axis_offset_tol)

        # ---------- cyl-plane ----------
        self.panel.add_fixed(8)
        self.panel.add_child(gui.Label("Cyl-Plane"))

        self.cp_enable_axis_plane_angle = gui.Checkbox("轴线-平面法向夹角约束")
        self.cp_enable_axis_plane_angle.checked = True

        self.cp_axis_plane_angle_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cp_axis_plane_angle_tol.set_limits(0.001, 30.0)
        self.cp_axis_plane_angle_tol.double_value = 0.10

        self.cp_enable_axis_plane_dist = gui.Checkbox("轴线到平面距离约束")
        self.cp_enable_axis_plane_dist.checked = True

        self.cp_target_axis_plane_dist = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cp_target_axis_plane_dist.set_limits(-10000.0, 10000.0)
        self.cp_target_axis_plane_dist.double_value = 0.0

        self.cp_axis_plane_dist_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cp_axis_plane_dist_tol.set_limits(0.001, 1000.0)
        self.cp_axis_plane_dist_tol.double_value = 0.10

        self.panel.add_child(self.cp_enable_axis_plane_angle)
        self.panel.add_child(gui.Label("角度容差 (deg)"))
        self.panel.add_child(self.cp_axis_plane_angle_tol)
        self.panel.add_child(self.cp_enable_axis_plane_dist)
        self.panel.add_child(gui.Label("目标轴线到平面距离 d (mm)"))
        self.panel.add_child(self.cp_target_axis_plane_dist)
        self.panel.add_child(gui.Label("距离容差 (mm)"))
        self.panel.add_child(self.cp_axis_plane_dist_tol)

        # ---------- 操作按钮 ----------
        self.panel.add_fixed(10)

        self.btn_add = gui.Button("Add Constraint")
        self.btn_add.set_on_clicked(self._on_add)
        self.panel.add_child(self.btn_add)

        self.btn_clear = gui.Button("Clear current")
        self.btn_clear.set_on_clicked(self._on_clear)
        self.panel.add_child(self.btn_clear)

        self.btn_remove_last = gui.Button("Remove Last")
        self.btn_remove_last.set_on_clicked(self._on_remove_last)
        self.panel.add_child(self.btn_remove_last)

        self.panel.add_fixed(10)
        self.panel.add_child(gui.Label("已添加约束"))

        self.lst = gui.ListView()
        self.lst.set_items([])
        self.panel.add_child(self.lst)

        self.panel.add_fixed(10)
        self.btn_done = gui.Button("Done")
        self.btn_done.set_on_clicked(self._on_done)
        self.panel.add_child(self.btn_done)

        self.tip = gui.Label("Tip: 左 CAD、中间扫描件；点击模型本体选择特征。")
        self.panel.add_child(self.tip)

        self._on_kind_changed("plane-plane", 0)

    def _on_kind_changed(self, text, index) -> None:
        """切换约束类型时，更新提示文本。"""
        kind_idx = int(index)
        if kind_idx == 0:
            self.tip.text = "当前为 plane-plane：请两边都选择平面；d 的方向由 CAD 法向决定。"
        elif kind_idx == 1:
            self.tip.text = "当前为 cyl-cyl：请两边都选择圆柱。"
        else:
            self.tip.text = "当前为 cyl-plane：左侧选圆柱，右侧选平面。"

    def _refresh_list(self) -> None:
        """刷新右侧约束列表显示。"""
        items = []

        for i, c in enumerate(self.constraints):
            if c.kind == "plane_plane":
                items.append(
                    f"{i + 1}. plane_plane CAD#{c.cad_id} ↔ SCAN#{c.scan_id} | "
                    f"d={c.target_gap_mm:.3f}±{c.gap_tol_mm:.3f}"
                )
            elif c.kind == "cyl_cyl":
                items.append(
                    f"{i + 1}. cyl_cyl CAD#{c.cad_id} ↔ SCAN#{c.scan_id} | "
                    f"off_tol={c.axis_offset_tol_mm:.3f}"
                )
            else:
                items.append(
                    f"{i + 1}. cyl_plane CAD#{c.cad_id} ↔ SCAN#{c.scan_id} | "
                    f"d={c.target_axis_plane_dist_mm:.3f}±{c.axis_plane_dist_tol_mm:.3f}"
                )

        self.lst.set_items(items)

    # =========================
    # 拾取逻辑
    # =========================

    def _cast_ray_hit(
        self,
        scene: o3d.t.geometry.RaycastingScene,
        widget: gui.SceneWidget,
        x: int,
        y: int,
    ):
        """从屏幕坐标发射射线，返回与模型的命中结果。"""
        cam = widget.scene.camera
        w = float(widget.frame.width)
        h = float(widget.frame.height)

        if w <= 1 or h <= 1:
            return None

        V = np.asarray(cam.get_view_matrix(), dtype=np.float64).reshape(4, 4)
        P = np.asarray(cam.get_projection_matrix(), dtype=np.float64).reshape(4, 4)
        inv_vp = np.linalg.inv(P @ V)

        def try_coords(xx: float, yy: float):
            if xx < 0 or yy < 0 or xx > w or yy > h:
                return None

            nx = (2.0 * xx / w) - 1.0
            ny = 1.0 - (2.0 * yy / h)

            p_near = inv_vp @ np.array([nx, ny, -1.0, 1.0], dtype=np.float64)
            p_far = inv_vp @ np.array([nx, ny, 1.0, 1.0], dtype=np.float64)

            p_near /= (p_near[3] + 1e-12)
            p_far /= (p_far[3] + 1e-12)

            origin = p_near[:3].astype(np.float32)
            direction = p_far[:3] - p_near[:3]
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-12:
                return None

            direction = (direction / direction_norm).astype(np.float32)

            rays = o3d.core.Tensor(
                [[origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]]],
                dtype=o3d.core.Dtype.Float32,
            )

            hit = scene.cast_rays(rays)
            gid = int(hit["geometry_ids"][0].item())
            t_hit = float(hit["t_hit"][0].item())

            if gid == scene.INVALID_ID or not np.isfinite(t_hit):
                return None

            primitive_id = None
            if "primitive_ids" in hit:
                primitive_id = int(hit["primitive_ids"][0].item())

            point = origin.astype(np.float64) + t_hit * direction.astype(np.float64)

            return {
                "gid": gid,
                "primitive_id": primitive_id,
                "t_hit": t_hit,
                "point": point,
            }

        result = try_coords(float(x), float(y))
        if result is not None:
            return result

        return try_coords(float(x) - float(widget.frame.x), float(y) - float(widget.frame.y))

    def _feature_from_triangle_map(
        self,
        side: str,
        primitive_id: Optional[int],
        preferred_kind: Optional[str] = None,
    ) -> Optional[FeatureInfo]:
        """优先通过 triangle_map 精确查找命中的特征。"""
        if primitive_id is None:
            return None

        maps = []
        if side == "cad":
            if preferred_kind in (None, "plane"):
                maps.append(("plane", self.cad_plane_triangle_map, self.cad_plane_feats))
            if preferred_kind in (None, "cyl"):
                maps.append(("cyl", self.cad_cyl_triangle_map, self.cad_cyl_feats))
        else:
            if preferred_kind in (None, "plane"):
                maps.append(("plane", self.scan_plane_triangle_map, self.scan_plane_feats))
            if preferred_kind in (None, "cyl"):
                maps.append(("cyl", self.scan_cyl_triangle_map, self.scan_cyl_feats))

        for _, feat_map, feat_dict in maps:
            for fid, tri_ids in feat_map.items():
                arr = np.asarray(tri_ids).reshape(-1)
                if arr.size > 0 and np.any(arr == primitive_id) and fid in feat_dict:
                    return feat_dict[fid]

        return None

    def _resolve_feature_from_hit(
        self,
        side: str,
        hit_point: np.ndarray,
        primitive_id: Optional[int],
        preferred_kind: Optional[str] = None,
    ) -> Optional[FeatureInfo]:
        """
        根据射线命中结果解析特征：
        1. 优先用 triangle_map 精确查找；
        2. 若失败，则按命中点到三角面中心的最近距离近似匹配。
        """
        mapped = self._feature_from_triangle_map(side, primitive_id, preferred_kind)
        if mapped is not None:
            return mapped

        feats = self.cad_all_feats if side == "cad" else self.scan_all_feats
        if preferred_kind is not None:
            feats = [f for f in feats if f.kind == preferred_kind]

        if not feats:
            return None

        best_feat = None
        best_score = float("inf")
        point = np.asarray(hit_point, dtype=np.float64).reshape(1, 3)

        for feat in feats:
            tri_centers = feat.triangle_centers
            if tri_centers.size == 0:
                continue

            d2 = np.sum((tri_centers - point) ** 2, axis=1)
            score = float(np.min(d2))
            if score < best_score:
                best_score = score
                best_feat = feat

        return best_feat

    def _current_expected_kind(self, side: str) -> Optional[str]:
        """根据当前约束类型，推断某一侧应该选择的特征类型。"""
        kind_idx = int(self.kind_combo.selected_index)
        if kind_idx == 0:
            return "plane"
        if kind_idx == 1:
            return "cyl"
        return "cyl" if side == "cad" else "plane"

    def _make_selection(self, feat: FeatureInfo) -> SelectionState:
        """由 FeatureInfo 生成当前选择状态。"""
        return SelectionState(
            kind=feat.kind,
            fid=feat.fid,
            center=feat.center.copy(),
            direction=feat.direction.copy(),
            feature=feat,
        )

    # =========================
    # 高亮显示
    # =========================

    def _remove_scene_selection(self, side: str) -> None:
        """移除某一侧场景中的高亮网格和方向箭头。"""
        scene = self.cad_scene.scene if side == "cad" else self.scan_scene.scene

        if side == "cad":
            if self._cad_hl_added:
                scene.remove_geometry(self._cad_hl_name)
                self._cad_hl_added = False
            if self._cad_arrow_added:
                scene.remove_geometry(self._cad_arrow_name)
                self._cad_arrow_added = False
        else:
            if self._scan_hl_added:
                scene.remove_geometry(self._scan_hl_name)
                self._scan_hl_added = False
            if self._scan_arrow_added:
                scene.remove_geometry(self._scan_arrow_name)
                self._scan_arrow_added = False

    def _make_arrow_mesh(
        self,
        center: np.ndarray,
        direction: np.ndarray,
        scale: float,
    ) -> o3d.geometry.TriangleMesh:
        """生成一个沿 direction 指向的箭头网格。"""
        scale = float(max(scale, 1.0))
        cyl_r = scale * 0.02
        cone_r = scale * 0.04
        cyl_h = scale * 0.70
        cone_h = scale * 0.30

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cyl_r,
            cone_radius=cone_r,
            cylinder_height=cyl_h,
            cone_height=cone_h,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )

        R = _rot_z_to_v(direction)
        arrow.rotate(R, center=np.zeros(3))
        arrow.translate(np.asarray(center, dtype=np.float64))
        arrow.compute_vertex_normals()
        return arrow

    def _highlight_selection(self, side: str, sel: SelectionState) -> None:
        """高亮显示当前选择的特征，并绘制方向箭头。"""
        self._remove_scene_selection(side)

        scene = self.cad_scene.scene if side == "cad" else self.scan_scene.scene
        hl_name = self._cad_hl_name if side == "cad" else self._scan_hl_name
        ar_name = self._cad_arrow_name if side == "cad" else self._scan_arrow_name

        hl_color = (1.0, 0.15, 0.15) if side == "cad" else (0.1, 0.55, 1.0)
        arr_color = (1.0, 0.3, 0.0) if side == "cad" else (0.0, 0.7, 0.6)

        mesh = self._get_feature_mesh_from_base(side, sel.kind, sel.fid)
        if mesh is None:
            mesh = copy.deepcopy(sel.feature.mesh)

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        scene.add_geometry(hl_name, mesh, self._mat_unlit(hl_color, 1.0))

        bbox = mesh.get_axis_aligned_bounding_box()
        diag = float(np.linalg.norm(bbox.get_extent()))
        arrow_scale = max(diag * 0.25, sel.feature.avg_radius * 0.8)
        arrow = self._make_arrow_mesh(sel.center, sel.direction, arrow_scale)

        scene.add_geometry(ar_name, arrow, self._mat_unlit(arr_color, 1.0))

        if side == "cad":
            self._cad_hl_added = True
            self._cad_arrow_added = True
        else:
            self._scan_hl_added = True
            self._scan_arrow_added = True

    # =========================
    # 鼠标事件
    # =========================

    def _on_mouse_cad(self, event):
        """处理 CAD 视图中的鼠标点击事件。"""
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and _is_left_click(event):
            hit = self._cast_ray_hit(self.ray_cad_base, self.cad_scene, event.x, event.y)
            if hit is None:
                self.tip.text = "CAD 侧没有命中模型。"
                return gui.Widget.EventCallbackResult.HANDLED

            feat = self._resolve_feature_from_hit(
                side="cad",
                hit_point=hit["point"],
                primitive_id=hit["primitive_id"],
                preferred_kind=self._current_expected_kind("cad"),
            )
            if feat is None:
                self.tip.text = "CAD 侧命中了模型，但没有解析到可用特征。"
                return gui.Widget.EventCallbackResult.HANDLED

            self.current_cad = self._make_selection(feat)
            self.lbl_cad.text = self._selection_label("CAD", self.current_cad)
            self._highlight_selection("cad", self.current_cad)
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse_scan(self, event):
        """处理扫描件视图中的鼠标点击事件。"""
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and _is_left_click(event):
            hit = self._cast_ray_hit(self.ray_scan_base, self.scan_scene, event.x, event.y)
            if hit is None:
                self.tip.text = "扫描件侧没有命中模型。"
                return gui.Widget.EventCallbackResult.HANDLED

            feat = self._resolve_feature_from_hit(
                side="scan",
                hit_point=hit["point"],
                primitive_id=hit["primitive_id"],
                preferred_kind=self._current_expected_kind("scan"),
            )
            if feat is None:
                self.tip.text = "扫描件侧命中了模型，但没有解析到可用特征。"
                return gui.Widget.EventCallbackResult.HANDLED

            self.current_scan = self._make_selection(feat)
            self.lbl_scan.text = self._selection_label("扫描件", self.current_scan)
            self._highlight_selection("scan", self.current_scan)
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _selection_label(self, side_name: str, sel: Optional[SelectionState]) -> str:
        """生成当前选择的文本描述。"""
        if sel is None:
            return f"当前 {side_name}: (none)"

        vec = np.array2string(sel.direction, precision=3, suppress_small=True)
        vec_name = "n" if sel.kind == "plane" else "axis"
        return f"当前 {side_name}: {sel.kind}#{sel.fid} {vec_name}={vec}"

    # =========================
    # 按钮动作
    # =========================

    def _on_clear(self) -> None:
        """清除当前 CAD / 扫描件 选择。"""
        self.current_cad = None
        self.current_scan = None
        self.lbl_cad.text = "当前 CAD: (none)"
        self.lbl_scan.text = "当前 扫描件: (none)"
        self._remove_scene_selection("cad")
        self._remove_scene_selection("scan")
        self.tip.text = "已清除当前选择。"

    def _on_add(self) -> None:
        """将当前选择转换为约束并加入列表。"""
        if self.current_cad is None or self.current_scan is None:
            self.tip.text = "需要同时选中 CAD 和 扫描件。"
            return

        kind_idx = int(self.kind_combo.selected_index)

        if kind_idx == 0:
            if self.current_cad.kind != "plane" or self.current_scan.kind != "plane":
                self.tip.text = "plane-plane 需要两边都选平面。"
                return

            spec = PairConstraintSpec(
                kind="plane_plane",
                cad_id=self.current_cad.fid,
                scan_id=self.current_scan.fid,
                enable_angle=bool(self.pl_enable_angle.checked),
                angle_tol_deg=float(self.pl_angle_tol.double_value),
                enable_gap=bool(self.pl_enable_gap.checked),
                target_gap_mm=float(self.pl_target_gap.double_value),
                gap_tol_mm=float(self.pl_gap_tol.double_value),
                gap_ref="cad_normal",
            )

        elif kind_idx == 1:
            if self.current_cad.kind != "cyl" or self.current_scan.kind != "cyl":
                self.tip.text = "cyl-cyl 需要两边都选圆柱。"
                return

            spec = PairConstraintSpec(
                kind="cyl_cyl",
                cad_id=self.current_cad.fid,
                scan_id=self.current_scan.fid,
                enable_axis_angle=bool(self.cy_enable_axis_angle.checked),
                axis_angle_tol_deg=float(self.cy_axis_angle_tol.double_value),
                enable_axis_offset=bool(self.cy_enable_axis_offset.checked),
                axis_offset_tol_mm=float(self.cy_axis_offset_tol.double_value),
            )

        else:
            if self.current_cad.kind != "cyl" or self.current_scan.kind != "plane":
                self.tip.text = "cyl-plane 需要 CAD 选圆柱，扫描件选平面。"
                return

            spec = PairConstraintSpec(
                kind="cyl_plane",
                cad_id=self.current_cad.fid,
                scan_id=self.current_scan.fid,
                enable_axis_plane_angle=bool(self.cp_enable_axis_plane_angle.checked),
                axis_plane_angle_tol_deg=float(self.cp_axis_plane_angle_tol.double_value),
                enable_axis_plane_dist=bool(self.cp_enable_axis_plane_dist.checked),
                target_axis_plane_dist_mm=float(self.cp_target_axis_plane_dist.double_value),
                axis_plane_dist_tol_mm=float(self.cp_axis_plane_dist_tol.double_value),
                axis_plane_dist_ref="plane_normal",
            )

        self.constraints.append(spec)
        self._refresh_list()
        self.tip.text = f"已添加约束 #{len(self.constraints)}"

    def _on_remove_last(self) -> None:
        """删除最后一条已添加约束。"""
        if not self.constraints:
            self.tip.text = "没有可删除的约束。"
            return

        self.constraints.pop()
        self._refresh_list()
        self.tip.text = "已删除最后一条约束。"

    def _on_done(self) -> None:
        """结束选择流程。"""
        if not self.constraints:
            self.tip.text = "至少添加一条约束再 Done。"
            return

        self.win.close()

    def run(self) -> list[PairConstraintSpec]:
        """运行 GUI 主循环，并返回最终约束列表。"""
        self.app.run()
        return self.constraints
