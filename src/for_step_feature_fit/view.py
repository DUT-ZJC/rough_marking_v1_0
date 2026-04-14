"""Open3D viewer for STEP-guided STL fitting results."""

from __future__ import annotations

import copy
import os
from typing import Callable

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .core_types import FaceFitResult, FitThresholds
from .pipeline import StepSTLFitSession


SURFACE_NAME_CN = {
    "plane": "平面",
    "cylinder": "柱面",
    "cone": "锥面",
    "sphere": "球面",
    "torus": "环面",
}

FEATURE_TYPE_ORDER = ["plane", "cylinder", "cone", "sphere", "torus"]


def _surface_name_cn(surface_type: str) -> str:
    """Return the Chinese display name of a supported surface type."""
    return SURFACE_NAME_CN.get(surface_type, surface_type)


def _make_slider(
    min_value: float,
    max_value: float,
    value: float,
    on_change: Callable[[float], None],
) -> gui.Slider:
    """Create a bound Open3D slider."""
    slider = gui.Slider(gui.Slider.DOUBLE)
    slider.set_limits(min_value, max_value)
    slider.double_value = float(value)
    slider.set_on_value_changed(on_change)
    return slider


def _outlier_points_from_result(
    session: StepSTLFitSession,
    result: FaceFitResult,
) -> o3d.geometry.PointCloud:
    """Convert outlier triangles into a point cloud of triangle centers."""
    cache = session.scan_cache
    if cache is None or result.outlier_triangles.size == 0:
        return o3d.geometry.PointCloud()

    points = cache.tri_centers[result.outlier_triangles]
    if len(points) == 0:
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = np.tile(np.array([[1.0, 0.92, 0.15]], dtype=np.float64), (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _is_left_click(event: gui.MouseEvent) -> bool:
    """Return True when the current mouse event is a left-button click."""
    try:
        if hasattr(event, "button") and event.button == gui.MouseButton.LEFT:
            return True
        if hasattr(event, "is_button_down"):
            return bool(event.is_button_down(gui.MouseButton.LEFT))
    except Exception:
        pass
    return False


class StepFitViewerApp:
    """Interactive viewer for localized STEP-driven STL fitting results."""

    def __init__(self, session: StepSTLFitSession, thresholds: FitThresholds) -> None:
        self.session = session
        self.thresholds = thresholds
        self.session.ensure_loaded()

        self.current_index = 0
        self.current_result: FaceFitResult | None = None
        self.all_results: list[FaceFitResult] = []
        self._face_result_cache: dict[tuple[int, tuple[float, float, float, float, int]], FaceFitResult] = {}
        self._merged_all_outliers_mesh: o3d.geometry.TriangleMesh | None = None
        self.type_to_indices = self._group_indices_by_surface_type()
        self.type_lists: dict[str, gui.ListView] = {}
        self._selection_syncing = False

        self.show_support = False
        self.show_step_face = True
        self.show_all_outliers = False
        self._scan_pick_scene: o3d.t.geometry.RaycastingScene | None = None

        self.app = gui.Application.instance
        self.app.initialize()
        self._configure_default_font()

        self.window = self.app.create_window("STEP解析面拟合与毛刺标注", 1750, 980)
        self.window.set_on_layout(self._on_layout)

        self.left_panel = gui.ScrollableVert(6, gui.Margins(10, 10, 10, 10))
        self.window.add_child(self.left_panel)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.97, 0.97, 0.97, 1.0])
        self.scene_widget.set_on_mouse(self._on_scene_mouse)
        self.window.add_child(self.scene_widget)

        self.right_panel = gui.Vert(6, gui.Margins(10, 10, 10, 10))
        self.window.add_child(self.right_panel)

        self._init_materials()
        self._build_left_panel()
        self._build_right_panel()
        self._add_static_scene()

        self._sync_grouped_selection(0)
        self._refresh_current_face()
        self.app.post_to_main_thread(self.window, self._setup_camera)

    def _configure_default_font(self) -> None:
        """Load a Windows CJK font so Chinese UI labels render correctly."""
        font_candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
        for font_path in font_candidates:
            if not os.path.exists(font_path):
                continue
            font = gui.FontDescription(font_path)
            font.add_typeface_for_language(font_path, "zh")
            self.app.set_font(self.app.DEFAULT_FONT_ID, font)
            return

    def _init_materials(self) -> None:
        """Initialize reusable rendering materials."""
        self.mat_scan = rendering.MaterialRecord()
        self.mat_scan.shader = "defaultLit"
        self.mat_scan.base_color = (0.62, 0.64, 0.67, 1.0)

        self.mat_step = rendering.MaterialRecord()
        self.mat_step.shader = "defaultLitTransparency"
        self.mat_step.base_color = (0.10, 0.35, 0.95, 0.38)

        self.mat_support = rendering.MaterialRecord()
        self.mat_support.shader = "defaultLitTransparency"
        self.mat_support.base_color = (0.10, 0.82, 0.95, 0.18)

        self.mat_inlier = rendering.MaterialRecord()
        self.mat_inlier.shader = "defaultLitTransparency"
        self.mat_inlier.base_color = (0.17, 0.72, 0.24, 0.95)

        self.mat_outlier = rendering.MaterialRecord()
        self.mat_outlier.shader = "defaultLitTransparency"
        self.mat_outlier.base_color = (0.98, 0.28, 0.10, 0.86)

        self.mat_outlier_points = rendering.MaterialRecord()
        self.mat_outlier_points.shader = "defaultUnlit"
        self.mat_outlier_points.base_color = (1.0, 0.92, 0.15, 1.0)
        self.mat_outlier_points.point_size = 8.0

    def _group_indices_by_surface_type(self) -> dict[str, list[int]]:
        """Group transformed STEP faces by type and sort each group by area."""
        grouped: dict[str, list[int]] = {}
        for face_index, face in enumerate(self.session.transformed_faces):
            grouped.setdefault(face.face.surface_type, []).append(face_index)
        for surface_type, indices in grouped.items():
            indices.sort(
                key=lambda idx: (
                    -float(self.session.transformed_faces[idx].face.area_mm2),
                    int(self.session.transformed_faces[idx].face.id),
                )
            )
        return grouped

    def _ordered_surface_types(self) -> list[str]:
        extras = sorted(t for t in self.type_to_indices.keys() if t not in FEATURE_TYPE_ORDER)
        return [t for t in FEATURE_TYPE_ORDER if t in self.type_to_indices] + extras

    def _group_list_labels(self, surface_type: str) -> list[str]:
        labels: list[str] = []
        for face_index in self.type_to_indices.get(surface_type, []):
            face = self.session.transformed_faces[face_index]
            labels.append(
                f"面 {face.face.id} [{_surface_name_cn(face.face.surface_type)}] "
                f"面积={face.face.area_mm2:.1f}"
            )
        return labels

    def _make_group_selection_handler(self, surface_type: str):
        def _handler(value: str, is_double_click: bool) -> None:
            self._on_face_selected(surface_type, value, is_double_click)

        return _handler

    def _sync_grouped_selection(self, global_index: int) -> None:
        self._selection_syncing = True
        try:
            for surface_type, list_view in self.type_lists.items():
                local_index = -1
                indices = self.type_to_indices.get(surface_type, [])
                for i, face_index in enumerate(indices):
                    if face_index == global_index:
                        local_index = i
                        break
                list_view.selected_index = local_index
        finally:
            self._selection_syncing = False

    def _set_current_index(self, face_index: int, source_text: str) -> None:
        self.current_index = int(face_index)
        self._sync_grouped_selection(self.current_index)
        face = self.session.transformed_faces[self.current_index].face
        self.lbl_pick_state.text = (
            f"点击状态：{source_text} 面 {face.id} / {_surface_name_cn(face.surface_type)}"
        )
        self._refresh_current_face()

    def _build_left_panel(self) -> None:
        """Build the feature list and threshold controls."""
        self.left_panel.add_child(gui.Label("特征列表"))
        self.lbl_paths = gui.Label(
            f"STEP: {self.session.step_path}\nSTL: {self.session.scan_stl_path}"
        )
        self.left_panel.add_child(self.lbl_paths)

        self.lbl_total = gui.Label(f"总特征数：{len(self.session.transformed_faces)}")
        self.left_panel.add_child(self.lbl_total)

        for surface_type in self._ordered_surface_types():
            indices = self.type_to_indices.get(surface_type, [])
            if not indices:
                continue
            section = gui.CollapsableVert(
                f"{_surface_name_cn(surface_type)}（{len(indices)}）",
                4,
                gui.Margins(6, 4, 6, 4),
            )
            list_view = gui.ListView()
            list_view.set_items(self._group_list_labels(surface_type))
            list_view.set_max_visible_items(min(max(len(indices), 3), 8))
            list_view.set_on_selection_changed(self._make_group_selection_handler(surface_type))
            section.add_child(list_view)
            self.left_panel.add_child(section)
            self.type_lists[surface_type] = list_view

        self.left_panel.add_fixed(8)
        self.left_panel.add_child(gui.Label("阈值设置（单位：mm）"))

        self.lbl_support = gui.Label("")
        self.sld_support = _make_slider(0.2, 8.0, self.thresholds.support_gap_mm, self._on_support_gap_changed)
        self.left_panel.add_child(self.lbl_support)
        self.left_panel.add_child(self.sld_support)

        self.lbl_plane = gui.Label("")
        self.sld_plane = _make_slider(0.05, 5.0, self.thresholds.plane_tol_mm, self._on_plane_tol_changed)
        self.left_panel.add_child(self.lbl_plane)
        self.left_panel.add_child(self.sld_plane)

        self.lbl_cyl = gui.Label("")
        self.sld_cyl = _make_slider(0.05, 5.0, self.thresholds.cylinder_tol_mm, self._on_cyl_tol_changed)
        self.left_panel.add_child(self.lbl_cyl)
        self.left_panel.add_child(self.sld_cyl)

        self.lbl_generic = gui.Label("")
        self.sld_generic = _make_slider(0.05, 6.0, self.thresholds.generic_tol_mm, self._on_generic_tol_changed)
        self.left_panel.add_child(self.lbl_generic)
        self.left_panel.add_child(self.sld_generic)

        self._sync_slider_labels()

        self.left_panel.add_fixed(6)
        self.chk_show_step = gui.Checkbox("显示STEP解析面")
        self.chk_show_step.checked = self.show_step_face
        self.chk_show_step.set_on_checked(self._on_show_step_checked)
        self.left_panel.add_child(self.chk_show_step)

        self.chk_show_support = gui.Checkbox("显示支撑区域")
        self.chk_show_support.checked = self.show_support
        self.chk_show_support.set_on_checked(self._on_show_support_checked)
        self.left_panel.add_child(self.chk_show_support)

        self.chk_show_all = gui.Checkbox("叠加全部毛刺")
        self.chk_show_all.checked = self.show_all_outliers
        self.chk_show_all.set_on_checked(self._on_show_all_checked)
        self.left_panel.add_child(self.chk_show_all)

        self.left_panel.add_fixed(8)
        self.btn_refresh = gui.Button("刷新当前特征")
        self.btn_refresh.set_on_clicked(self._refresh_current_face)
        self.left_panel.add_child(self.btn_refresh)

        self.btn_all = gui.Button("统计全部毛刺")
        self.btn_all.set_on_clicked(self._analyze_all_faces)
        self.left_panel.add_child(self.btn_all)

        self.lbl_global = gui.Label("全部毛刺统计：尚未计算")
        self.left_panel.add_child(self.lbl_global)

    def _build_right_panel(self) -> None:
        """Build the information panel for the current feature."""
        self.right_panel.add_child(gui.Label("当前选中特征"))

        self.lbl_pick_hint = gui.Label("操作：左键点击模型，可直接选中最近的解析面/曲面。")
        self.lbl_pick_state = gui.Label("点击状态：未选中")
        self.lbl_legend = gui.Label(
            "颜色说明：蓝=STEP解析面，绿=拟合通过，橙红+黄点=毛刺/异常，灰=原始STL"
        )

        self.lbl_type = gui.Label("类型：-")
        self.lbl_status = gui.Label("状态：-")
        self.lbl_area = gui.Label("STEP面积：-")
        self.lbl_support_area = gui.Label("支撑面积：-")
        self.lbl_inlier_area = gui.Label("拟合通过面积：-")
        self.lbl_outlier_area = gui.Label("毛刺/异常面积：-")
        self.lbl_ratio = gui.Label("通过比例：-")
        self.lbl_resid = gui.Label("残差：-")
        self.lbl_reg = gui.Label(f"配准信息：{self.session.registration_info}")

        for widget in [
            self.lbl_pick_hint,
            self.lbl_pick_state,
            self.lbl_legend,
            self.lbl_type,
            self.lbl_status,
            self.lbl_area,
            self.lbl_support_area,
            self.lbl_inlier_area,
            self.lbl_outlier_area,
            self.lbl_ratio,
            self.lbl_resid,
            self.lbl_reg,
        ]:
            self.right_panel.add_child(widget)

    def _sync_slider_labels(self) -> None:
        self.lbl_support.text = f"局部支撑搜索距离：{self.thresholds.support_gap_mm:.2f}"
        self.lbl_plane.text = f"平面拟合容差：{self.thresholds.plane_tol_mm:.2f}"
        self.lbl_cyl.text = f"柱面拟合容差：{self.thresholds.cylinder_tol_mm:.2f}"
        self.lbl_generic.text = f"其他曲面容差：{self.thresholds.generic_tol_mm:.2f}"

    def _threshold_cache_key(self) -> tuple[float, float, float, float, int]:
        """Create a stable cache key for the current threshold state."""
        return (
            round(float(self.thresholds.support_gap_mm), 4),
            round(float(self.thresholds.plane_tol_mm), 4),
            round(float(self.thresholds.cylinder_tol_mm), 4),
            round(float(self.thresholds.generic_tol_mm), 4),
            int(self.thresholds.min_support_triangles),
        )

    def _invalidate_analysis_cache(self) -> None:
        """Drop cached fitting results when thresholds change."""
        self._face_result_cache.clear()
        self.all_results = []
        self._merged_all_outliers_mesh = None
        self.show_all_outliers = False
        if hasattr(self, "chk_show_all"):
            self.chk_show_all.checked = False

    def _add_static_scene(self) -> None:
        scan_mesh = copy.deepcopy(self.session.scan_mesh)
        if scan_mesh is not None and len(scan_mesh.triangles) > 0:
            if not scan_mesh.has_vertex_normals():
                scan_mesh.compute_vertex_normals()
            self.scene_widget.scene.add_geometry("scan_mesh", scan_mesh, self.mat_scan)

    def _clear_dynamic_geometries(self) -> None:
        for name in [
            "step_face",
            "support_mesh",
            "inlier_mesh",
            "outlier_mesh",
            "outlier_points",
            "all_outliers",
        ]:
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.remove_geometry(name)

    def _add_mesh_if_nonempty(
        self,
        name: str,
        mesh: o3d.geometry.TriangleMesh,
        material: rendering.MaterialRecord,
    ) -> None:
        if mesh is None or len(mesh.triangles) == 0:
            return
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        self.scene_widget.scene.add_geometry(name, mesh, material)

    def _add_point_cloud_if_nonempty(
        self,
        name: str,
        pcd: o3d.geometry.PointCloud,
        material: rendering.MaterialRecord,
    ) -> None:
        if pcd is None or len(pcd.points) == 0:
            return
        self.scene_widget.scene.add_geometry(name, pcd, material)

    def _refresh_current_face(self) -> None:
        """Recompute and redraw the currently selected feature."""
        if self.show_all_outliers and len(self.all_results) == len(self.session.transformed_faces):
            self.current_result = self.all_results[self.current_index]
        else:
            cache_key = (int(self.current_index), self._threshold_cache_key())
            if cache_key not in self._face_result_cache:
                self._face_result_cache[cache_key] = self.session.analyze_face(self.current_index, self.thresholds)
            self.current_result = self._face_result_cache[cache_key]
        self._clear_dynamic_geometries()

        if self.show_all_outliers and self.all_results:
            if self._merged_all_outliers_mesh is None:
                self._merged_all_outliers_mesh = self._merged_outlier_mesh(self.all_results)
            self._add_mesh_if_nonempty("all_outliers", self._merged_all_outliers_mesh, self.mat_outlier)

        assert self.current_result is not None

        if self.show_step_face:
            self._add_mesh_if_nonempty(
                "step_face",
                self.current_result.transformed_face_mesh,
                self.mat_step,
            )
        if self.show_support:
            self._add_mesh_if_nonempty(
                "support_mesh",
                self.current_result.support_mesh,
                self.mat_support,
            )

        self._add_mesh_if_nonempty("inlier_mesh", self.current_result.inlier_mesh, self.mat_inlier)
        self._add_mesh_if_nonempty("outlier_mesh", self.current_result.outlier_mesh, self.mat_outlier)
        self._add_point_cloud_if_nonempty(
            "outlier_points",
            _outlier_points_from_result(self.session, self.current_result),
            self.mat_outlier_points,
        )
        self._update_info_panel()

    def _merged_outlier_mesh(self, results: list[FaceFitResult]) -> o3d.geometry.TriangleMesh:
        merged = o3d.geometry.TriangleMesh()
        for result in results:
            if len(result.outlier_mesh.triangles) > 0:
                merged += result.outlier_mesh
        if len(merged.triangles) > 0:
            merged.remove_duplicated_vertices()
            merged.remove_duplicated_triangles()
            merged.compute_vertex_normals()
        return merged

    def _analyze_all_faces(self) -> None:
        """Run the full global fitting pass for all STEP faces."""
        self.all_results = self.session.analyze_all_faces(self.thresholds)
        self._merged_all_outliers_mesh = None
        threshold_key = self._threshold_cache_key()
        self._face_result_cache.update(
            {
                (face_index, threshold_key): result
                for face_index, result in enumerate(self.all_results)
            }
        )
        self.show_all_outliers = True
        self.chk_show_all.checked = True

        total_outlier_area = sum(result.outlier_area_mm2 for result in self.all_results)
        total_support_area = sum(result.support_area_mm2 for result in self.all_results)
        self.lbl_global.text = (
            f"全部毛刺统计：毛刺面积={total_outlier_area:.1f} / 支撑面积={total_support_area:.1f}"
        )
        self._refresh_current_face()

    def _update_info_panel(self) -> None:
        result = self.current_result
        face = self.session.transformed_faces[self.current_index].face
        assert result is not None

        self.lbl_type.text = f"类型：{_surface_name_cn(face.surface_type)} | 面ID：{face.id}"
        self.lbl_status.text = f"状态：{result.status} | {result.message}"
        self.lbl_area.text = f"STEP面积：{face.area_mm2:.2f} mm^2"
        self.lbl_support_area.text = f"支撑面积：{result.support_area_mm2:.2f} mm^2"
        self.lbl_inlier_area.text = f"拟合通过面积：{result.inlier_area_mm2:.2f} mm^2"
        self.lbl_outlier_area.text = f"毛刺/异常面积：{result.outlier_area_mm2:.2f} mm^2"
        self.lbl_ratio.text = (
            f"通过比例：{result.inlier_ratio:.3f} | 支撑三角片数={len(result.support_triangles)}"
        )
        self.lbl_resid.text = (
            f"残差 mean/p95/max：{result.residual_mean_mm:.3f} / "
            f"{result.residual_p95_mm:.3f} / {result.residual_max_mm:.3f} mm"
        )

    def _ensure_scan_pick_scene(self) -> o3d.t.geometry.RaycastingScene:
        """Create the raycasting scene used for STL mouse picking."""
        if self._scan_pick_scene is None:
            assert self.session.scan_mesh is not None
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.session.scan_mesh))
            self._scan_pick_scene = scene
        return self._scan_pick_scene

    def _pick_scan_hit_from_mouse(self, event: gui.MouseEvent) -> tuple[np.ndarray, int] | None:
        """Cast a ray from the current mouse position to the STL scan mesh."""
        frame = self.scene_widget.frame
        width = float(frame.width)
        height = float(frame.height)
        if width <= 1 or height <= 1:
            return None

        local_x = float(event.x - frame.x)
        local_y = float(event.y - frame.y)
        if local_x < 0 or local_x >= width or local_y < 0 or local_y >= height:
            return None

        camera = self.scene_widget.scene.camera
        view = np.asarray(camera.get_view_matrix(), dtype=np.float64).reshape(4, 4)
        proj = np.asarray(camera.get_projection_matrix(), dtype=np.float64).reshape(4, 4)
        inv_vp = np.linalg.inv(proj @ view)

        ndc_x = (2.0 * local_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * local_y / height)

        p_near = inv_vp @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
        p_far = inv_vp @ np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64)
        p_near /= (p_near[3] + 1e-12)
        p_far /= (p_far[3] + 1e-12)

        direction = p_far[:3] - p_near[:3]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-12:
            return None
        direction /= direction_norm

        ray = np.concatenate([p_near[:3], direction]).astype(np.float32)[None, :]
        hit = self._ensure_scan_pick_scene().cast_rays(
            o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
        )
        t_hit = float(hit["t_hit"][0].item())
        if not np.isfinite(t_hit):
            return None
        primitive_id = int(hit["primitive_ids"][0].item()) if "primitive_ids" in hit else -1
        return p_near[:3] + direction * t_hit, primitive_id

    def _on_scene_mouse(self, event: gui.MouseEvent):
        """Handle left-click picking on the STL scene."""
        if event.type != gui.MouseEvent.BUTTON_DOWN:
            return gui.SceneWidget.EventCallbackResult.IGNORED
        if not _is_left_click(event):
            return gui.SceneWidget.EventCallbackResult.IGNORED

        picked = self._pick_scan_hit_from_mouse(event)
        if picked is None:
            self.lbl_pick_state.text = "点击状态：未命中扫描网格"
            return gui.SceneWidget.EventCallbackResult.IGNORED

        hit_point, triangle_id = picked
        pick_distance = max(2.0, float(self.thresholds.support_gap_mm) * 1.5)
        face_index = None
        if triangle_id >= 0:
            face_index = self.session.pick_face_by_triangle(
                triangle_id=triangle_id,
                point_world=hit_point,
                thresholds=self.thresholds,
                max_distance_mm=pick_distance,
            )
        if face_index is None:
            face_index = self.session.pick_face_by_point(hit_point, max_distance_mm=pick_distance)
        if face_index is None:
            self.lbl_pick_state.text = "点击状态：未找到附近解析面"
            return gui.SceneWidget.EventCallbackResult.IGNORED

        self._set_current_index(int(face_index), "点击选中")
        return gui.SceneWidget.EventCallbackResult.HANDLED

    def _on_face_selected(self, surface_type: str, value: str, is_double_click: bool) -> None:
        if self._selection_syncing:
            return
        if not value:
            return
        list_view = self.type_lists.get(surface_type)
        if list_view is None:
            return
        local_index = list_view.selected_index
        if local_index < 0:
            return
        global_indices = self.type_to_indices.get(surface_type, [])
        if local_index >= len(global_indices):
            return
        self._set_current_index(global_indices[local_index], "列表选中")

    def _on_support_gap_changed(self, value: float) -> None:
        self.thresholds.support_gap_mm = float(value)
        self._invalidate_analysis_cache()
        self._sync_slider_labels()
        self._refresh_current_face()

    def _on_plane_tol_changed(self, value: float) -> None:
        self.thresholds.plane_tol_mm = float(value)
        self._invalidate_analysis_cache()
        self._sync_slider_labels()
        self._refresh_current_face()

    def _on_cyl_tol_changed(self, value: float) -> None:
        self.thresholds.cylinder_tol_mm = float(value)
        self._invalidate_analysis_cache()
        self._sync_slider_labels()
        self._refresh_current_face()

    def _on_generic_tol_changed(self, value: float) -> None:
        self.thresholds.generic_tol_mm = float(value)
        self._invalidate_analysis_cache()
        self._sync_slider_labels()
        self._refresh_current_face()

    def _on_show_step_checked(self, checked: bool) -> None:
        self.show_step_face = bool(checked)
        self._refresh_current_face()

    def _on_show_support_checked(self, checked: bool) -> None:
        self.show_support = bool(checked)
        self._refresh_current_face()

    def _on_show_all_checked(self, checked: bool) -> None:
        self.show_all_outliers = bool(checked)
        self._refresh_current_face()

    def _setup_camera(self) -> None:
        """Initialize the scene camera on the STL bounding box."""
        assert self.session.scan_mesh is not None
        bbox = self.session.scan_mesh.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    def _on_layout(self, ctx) -> None:
        rect = self.window.content_rect
        left_w = 340
        right_w = 360
        scene_w = rect.width - left_w - right_w
        self.left_panel.frame = gui.Rect(rect.x, rect.y, left_w, rect.height)
        self.scene_widget.frame = gui.Rect(rect.x + left_w, rect.y, scene_w, rect.height)
        self.right_panel.frame = gui.Rect(rect.x + rect.width - right_w, rect.y, right_w, rect.height)

    def run(self) -> None:
        """Enter the Open3D GUI event loop."""
        self.app.run()
