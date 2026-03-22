import os
import sys
import platform
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import re

# ==========================================
# 终极导包解决方案：强制将项目根目录加入环境
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))      # 当前是 for_stl_feature 文件夹
project_root = os.path.dirname(os.path.dirname(current_dir))  # 退两级，回到 rough_marking_v1_0 根目录

if project_root not in sys.path:
    # 插入到最前，确保优先加载此项目的 core_types
    sys.path.insert(0, project_root)

try:
    from src.for_stl_feature.core_types import ScanPlaneFeature, ScanCylinderFeature
except ImportError:
    print("❌ 错误：无法导入 src.for_stl_feature.core_types。")
    print(f"当前路径: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


# ==========================================
# 定义通用接口：不管以后用什么算法，必须实现此格式
# ==========================================
class STLFeaturePipelineInterface:
    @staticmethod
    def extract(stl_path: str, **kwargs) -> tuple:
        """
        子类必须实现此函数。
        """
        raise NotImplementedError


# ==========================================
# 算法适配器：把提取管线适配到通用接口
# ==========================================
class MyAlgorithmAdapter(STLFeaturePipelineInterface):
    @staticmethod
    def extract(stl_path: str, **kwargs):  # ✨ 这里加上 **kwargs
        # 延迟导入 pipeline，防止循环依赖
        # (注意：如果你的文件名之前改成了 stl_extractor.py，这里就保持 stl_extractor)
        from src.for_stl_feature.stl_extractor import process_scan_features
        
        # ✨ 将所有额外参数 (如 method="ransac") 透传给底层处理函数
        return process_scan_features(stl_path, **kwargs)


# ==========================================
# 通用 UI 辅助函数
# ==========================================
def _setup_open3d_font() -> None:
    """配置中文字体以支持 GUI 显示"""
    system = platform.system()
    if system == "Darwin":
        hanzi = "STHeiti Light"
    elif system == "Windows":
        hanzi = r"c:/windows/fonts/msyh.ttc"
    else:
        hanzi = "NotoSansCJK"

    font = gui.FontDescription()
    try:
        font.add_typeface_for_language(hanzi, "zh_all")
        gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)
    except Exception as e:
        print(f"警告: 无法加载中文字体 ({e})，界面可能显示方块。")


def _is_left_click(event) -> bool:
    """判断事件是否为鼠标左键点击"""
    try:
        if hasattr(event, "button") and event.button == gui.MouseButton.LEFT:
            return True
        if hasattr(event, "is_button_down"):
            return bool(event.is_button_down(gui.MouseButton.LEFT))
    except Exception:
        pass
    return False


class FeatureViewerApp:
    def __init__(self, planes: list, cyls: list, remaining_mask: np.ndarray, base_mesh: o3d.geometry.TriangleMesh):
        self.planes = planes
        self.cyls = cyls
        self.remaining_mask = remaining_mask
        self.base_mesh = base_mesh

        # 生成“剩余网格” (用于可视化)
        self.remaining_mesh = self._generate_remaining_mesh()

        # 状态追踪
        self.current_selection = None # (type, id)

        # 映射表： primitive_id -> feature object (用于鼠标)
        self.prim_feature_map = {}
        # 映射表: (type, id) -> feature object (用于列表)
        self.id_feature_map = {"plane": {}, "cyl": {}}
        self._build_maps()

        # GUI 初始化
        self.app = gui.Application.instance
        self.app.initialize()
        _setup_open3d_font()

        # 设置材质
        mat_unlit = rendering.MaterialRecord()
        mat_unlit.shader = "defaultUnlit"
        self.mat_plane = rendering.MaterialRecord()
        self.mat_plane.shader = "defaultLit"
        self.mat_plane.base_color = (0.2, 0.8, 0.3, 1.0) # 绿色 overlay
        self.mat_cyl = rendering.MaterialRecord()
        self.mat_cyl.shader = "defaultLit"
        self.mat_cyl.base_color = (0.2, 0.4, 0.9, 1.0) # 蓝色 overlay
        self.mat_remain = rendering.MaterialRecord()
        self.mat_remain.shader = "defaultLit"
        self.mat_remain.base_color = (0.5, 0.5, 0.5, 0.3) # 半透明灰色
        self.mat_high = rendering.MaterialRecord()
        self.mat_high.shader = "defaultUnlit"
        self.mat_high.base_color = (1.0, 0.0, 0.0, 1.0) # 纯红高亮

        self.win = self.app.create_window("特征可视化与拾取工具", 1600, 900)
        self.win.set_on_layout(self._on_layout)

        # 1. 左侧面板 (列表)
        self.left_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.win.add_child(self.left_panel)
        self._build_left_panel()

        # 2. 中间 3D 场景
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.win.renderer)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.win.add_child(self.scene_widget)

        # 3. 右侧面板 (信息)
        self.right_panel = gui.Vert(0, gui.Margins(15, 15, 15, 15))
        self.win.add_child(self.right_panel)
        self._build_right_panel()

        # 射线求交场景 (用于鼠标拾取 base_mesh)
        self.ray_scene = o3d.t.geometry.RaycastingScene()
        if not self.base_mesh.has_vertex_normals():
            self.base_mesh.compute_vertex_normals()
        self.ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.base_mesh))

        # 渲染初始几何体
        self._add_geometries()
        self.app.post_to_main_thread(self.win, self._setup_camera)

        # 绑定鼠标事件
        self.scene_widget.set_on_mouse(self._on_mouse)

    def _build_maps(self):
        # 排序特征并建立 ID 映射 (用于列表)
        sorted_planes = sorted(self.planes, key=lambda x: x.id)
        sorted_cyls = sorted(self.cyls, key=lambda x: x.id)
        
        for p in sorted_planes:
            self.id_feature_map["plane"][p.id] = p
            # primitive id 映射 (用于鼠标)
            for tri_id in p.tri_indices:
                self.prim_feature_map[tri_id] = ("plane", p.id)
                
        for c in sorted_cyls:
            self.id_feature_map["cyl"][c.id] = c
            for tri_id in c.tri_indices:
                self.prim_feature_map[tri_id] = ("cyl", c.id)

    def _generate_remaining_mesh(self):
        if self.remaining_mask is None or np.sum(self.remaining_mask) == 0:
            return None
        # 复制完整网格，然后根据 mask 移除三角形
        mesh = o3d.geometry.TriangleMesh(self.base_mesh)
        # 移除 Mask=False 的三角形，保留 True
        mesh.remove_triangles_by_mask(~self.remaining_mask)
        mesh.compute_vertex_normals()
        return mesh

    def _add_geometries(self):
        # 1. 剩余部分 (半透明)
        if self.remaining_mesh:
            self.scene_widget.scene.add_geometry("remain_mesh", self.remaining_mesh, self.mat_remain)

        # 2. 平面 overlay (绿)
        for p_id, p in self.id_feature_map["plane"].items():
            name = f"plane_{p_id}"
            if not p.mesh.has_vertex_normals():
                p.mesh.compute_vertex_normals()
            self.scene_widget.scene.add_geometry(name, p.mesh, self.mat_plane)

        # 3. 圆柱 overlay (蓝)
        for c_id, c in self.id_feature_map["cyl"].items():
            name = f"cyl_{c_id}"
            if not c.mesh.has_vertex_normals():
                c.mesh.compute_vertex_normals()
            self.scene_widget.scene.add_geometry(name, c.mesh, self.mat_cyl)

    def _setup_camera(self):
        """相机居中对齐模型"""
        bbox = self.base_mesh.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    def _on_layout(self, ctx):
        """处理窗口自适应布局"""
        rect = self.win.content_rect
        left_w = 280
        right_w = 300
        scene_w = rect.width - left_w - right_w
        
        self.left_panel.frame = gui.Rect(rect.x, rect.y, left_w, rect.height)
        self.scene_widget.frame = gui.Rect(rect.x + left_w, rect.y, scene_w, rect.height)
        self.right_panel.frame = gui.Rect(rect.x + rect.width - right_w, rect.y, right_w, rect.height)

    def _build_left_panel(self):
        self.left_panel.add_child(gui.Label("--- 特征列表 ---"))
        self.left_panel.add_fixed(10)
        
        # 1. 平面列表
        self.left_panel.add_child(gui.Label("平面 (Planes)"))
        self.lv_planes = gui.ListView()
        self.lv_planes.set_items([f"Plane {p_id}" for p_id in sorted(self.id_feature_map["plane"].keys())])
        # 绑定列表选择事件
        self.lv_planes.set_on_selection_changed(self._on_list_planes_selected)
        self.left_panel.add_child(self.lv_planes)
        self.left_panel.add_fixed(15)

        # 2. 圆柱列表
        self.left_panel.add_child(gui.Label("圆柱 (Cylinders)"))
        self.lv_cyls = gui.ListView()
        self.lv_cyls.set_items([f"Cylinder {c_id}" for c_id in sorted(self.id_feature_map["cyl"].keys())])
        # 绑定列表选择事件
        self.lv_cyls.set_on_selection_changed(self._on_list_cyls_selected)
        self.left_panel.add_child(self.lv_cyls)

    def _build_right_panel(self):
        """构建右侧属性显示面板"""
        self.right_panel.add_child(gui.Label("--- 当前选中特征 ---"))
        self.lbl_type = gui.Label("类型: 无")
        self.lbl_id = gui.Label("ID: -")
        self.lbl_rmse = gui.Label("RMSE: -")
        self.lbl_param1 = gui.Label("面积/半径: -")
        self.lbl_param2 = gui.Label("法向/轴向: -")
        
        self.right_panel.add_child(self.lbl_type)
        self.right_panel.add_child(self.lbl_id)
        self.right_panel.add_child(self.lbl_rmse)
        self.right_panel.add_child(self.lbl_param1)
        self.right_panel.add_child(self.lbl_param2)

    # ==========================================
    # 特征拾取与高亮核心逻辑 (修复 Z-fighting，实现“纯红”)
    # ==========================================
    def _handle_pick_internal(self, feat_type: str, feat_id: int, from_mouse: bool = False):
        """统一的特征拾取处理接口，不管来自鼠标还是列表"""
        if self.current_selection == (feat_type, feat_id):
            return

        # 1. 恢复上一次选中的颜色
        if self.current_selection:
            prev_type, prev_id = self.current_selection
            prev_name = f"{prev_type}_{prev_id}"
            if self.scene_widget.scene.has_geometry(prev_name):
                # 如果 prev 被手动删除了（比如为了换红颜色），需要 re-add。
                # 现在的逻辑是只换材质，不删几何，所以 has_geometry 必为 True
                pass
            
            # 材质换回原色
            mat = self.mat_plane if prev_type == "plane" else self.mat_cyl
            self.scene_widget.scene.modify_geometry_material(prev_name, mat)

        # 2. 处理新的选中对象
        self.current_selection = (feat_type, feat_id)
        feat = self.id_feature_map[feat_type][feat_id]
        new_name = f"{feat_type}_{feat_id}"
        
        if not self.scene_widget.scene.has_geometry(new_name):
            return # 应该不会发生
            
        # ✨ 关键：为了不通过 scale(1.002) 引起 Z-fighting，
        # ✨ 我们直接将原有 Geometry 的材质改为 Unlit Vivid Red!
        # ✨ 这要求材质不支持光照(defaultUnlit)，从而保证看起来就是“纯红”。
        self.scene_widget.scene.modify_geometry_material(new_name, self.mat_high)

        # 3. 更新右侧信息面板
        self._update_right_panel(feat_type, feat)

        # 4. 同步更新左侧列表选择 (如果是鼠标点的场景)
        if from_mouse:
            self._sync_lists_to_mouse(feat_type, feat_id)

    def _clear_selection(self):
        if self.current_selection:
            prev_type, prev_id = self.current_selection
            prev_name = f"{prev_type}_{prev_id}"
            # 材质换回原色
            mat = self.mat_plane if prev_type == "plane" else self.mat_cyl
            self.scene_widget.scene.modify_geometry_material(prev_name, mat)
            self.current_selection = None
            
            # 清空右侧
            self.lbl_type.text = "类型: 无 (点到了背景)"
            self.lbl_id.text = "ID: -"
            self.lbl_rmse.text = "RMSE: -"
            self.lbl_param1.text = "面积/半径: -"
            self.lbl_param2.text = "法向/轴向: -"
            
            # 清空列表选择 (防止死循环)
            self.lv_planes.selected_index = -1
            self.lv_cyls.selected_index = -1

    def _update_right_panel(self, feat_type: str, feat):
        """更新面板文字"""
        self.lbl_type.text = f"类型: {'平面' if feat_type == 'plane' else '圆柱'}"
        self.lbl_id.text = f"ID: {feat.id}"
        self.lbl_rmse.text = f"RMSE: {feat.rmse:.4f} mm"
        
        if feat_type == 'plane':
            self.lbl_param1.text = f"面积: {feat.area:.2f} mm²"
            vec = np.array2string(feat.normal, precision=3, suppress_small=True)
            self.lbl_param2.text = f"法向: {vec}"
        else:
            self.lbl_param1.text = f"半径: {feat.radius:.3f} mm"
            vec = np.array2string(feat.axis_dir, precision=3, suppress_small=True)
            self.lbl_param2.text = f"轴向: {vec}"

    def _sync_lists_to_mouse(self, feat_type: str, feat_id: int):
        # 注意：这里需要暂时禁用事件处理，防止反向同步
        if feat_type == "plane":
            idx = sorted(self.id_feature_map["plane"].keys()).index(feat_id)
            self.lv_planes.selected_index = idx
            self.lv_cyls.selected_index = -1
        else:
            idx = sorted(self.id_feature_map["cyl"].keys()).index(feat_id)
            self.lv_cyls.selected_index = idx
            self.lv_planes.selected_index = -1

    # ==========================================
    # 场景鼠标交互逻辑
    # ==========================================
   # ==========================================
    # 场景鼠标交互逻辑
    # ==========================================
    def _on_mouse(self, event):
        """鼠标射线拾取逻辑"""
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and _is_left_click(event):
            # 获取相机矩阵
            cam = self.scene_widget.scene.camera
            w, h = float(self.scene_widget.frame.width), float(self.scene_widget.frame.height)
            
            V = np.asarray(cam.get_view_matrix()).reshape(4, 4)
            P = np.asarray(cam.get_projection_matrix()).reshape(4, 4)
            inv_vp = np.linalg.inv(P @ V)

            # ✨ 关键修复：减去 SceneWidget 的左上角偏移量，获取真实的局部相对坐标！
            local_x = event.x - self.scene_widget.frame.x
            local_y = event.y - self.scene_widget.frame.y

            # 屏幕坐标转 NDC
            nx = (2.0 * local_x / w) - 1.0
            ny = 1.0 - (2.0 * local_y / h)

            p_near = inv_vp @ np.array([nx, ny, -1.0, 1.0])
            p_far = inv_vp @ np.array([nx, ny, 1.0, 1.0])
            
            # 加入 epsilon 防止除以 0
            p_near /= (p_near[3] + 1e-12)
            p_far /= (p_far[3] + 1e-12)

            direction = p_far[:3] - p_near[:3]
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                return gui.Widget.EventCallbackResult.HANDLED
            direction /= dir_norm

            # 发射射线
            rays = o3d.core.Tensor([[*p_near[:3], *direction]], dtype=o3d.core.Dtype.Float32)
            hit = self.ray_scene.cast_rays(rays)
            
            t_hit = float(hit["t_hit"][0].item())
            if np.isfinite(t_hit) and "primitive_ids" in hit:
                prim_id = int(hit["primitive_ids"][0].item())
                # 命中底层 Triangle 网格，在 prim_feature_map 中查找是哪个特征
                if prim_id in self.prim_feature_map:
                    feat_type, feat_id = self.prim_feature_map[prim_id]
                    self._handle_pick_internal(feat_type, feat_id, from_mouse=True)
                else:
                    self._clear_selection()
            else:
                self._clear_selection()
            return gui.Widget.EventCallbackResult.HANDLED
            
        return gui.Widget.EventCallbackResult.IGNORED

    # ==========================================
    # 列表选择交互逻辑
    # ==========================================
    def _on_list_planes_selected(self, new_val: str, is_double_click: bool):
        if not new_val: return
        # 防止反向回调死循环
        if self.current_selection and self.current_selection[0] == "plane" and self.lv_planes.selected_index == -1:
            return

        # 提取 ID (从 "Plane 12" -> 12)
        match = re.search(r'\d+', new_val)
        if match:
            feat_id = int(match.group())
            # 调用统一接口 (不更新 mouse，因为 mouse 永远看 base mesh)
            self._handle_pick_internal("plane", feat_id, from_mouse=False)
            # 清空 cyls 的选择
            self.lv_cyls.selected_index = -1

    def _on_list_cyls_selected(self, new_val: str, is_double_click: bool):
        if not new_val: return
        if self.current_selection and self.current_selection[0] == "cyl" and self.lv_cyls.selected_index == -1:
            return

        match = re.search(r'\d+', new_val)
        if match:
            feat_id = int(match.group())
            self._handle_pick_internal("cyl", feat_id, from_mouse=False)
            # 清空 planes 的选择
            self.lv_planes.selected_index = -1

    def run(self):
        self.app.run()


# ==========================================
# 独立测试区域
# ==========================================
if __name__ == "__main__":
    print("=== 特征检视工具独立测试 ===")
    
    # 替换为你自己电脑上的测试 STL 路径
    test_stl = input("请输入测试用的 STL 文件路径 (或直接回车使用默认路径): ").strip()
    if not test_stl:
        test_stl = r"./data/2.stl"  # 填你常用的测试文件路径
        
    if not os.path.exists(test_stl):
        print(f"❌ 找不到文件: {test_stl}")
        sys.exit(1)
        
    print(f"开始提取 {test_stl} 的特征，请稍候...")
    
    try:
        # 1. 调用适配器提取数据
        # 这个架构允许以后随时切换新的策略，不需要改 Viewer 代码
        planes, cyls, mask, mesh = MyAlgorithmAdapter.extract(test_stl)
        """
        planes, cyls, mask, mesh = MyAlgorithmAdapter.extract(
            test_stl, 
            method="ransac",             # ✨ 必须改成你刚刚在注册表里起的名字！
            resolution_mm=1.3,         # ✅ 保留：用于指导 C++ 进行连通域聚类 (非常重要)
            plane_dist_tol=1.0,        # ✨ 距离容差 (epsilon)：点到平面的最大允许距离(mm)
            plane_angle_deg=6.0,       # ✨ 角度容差：法向量允许的最大偏差角度
            plane_min_area=100.0,      # ✨ 面积阈值：太小的碎面片直接扔掉
        )
        """
        print("提取成功！正在启动可视化界面...")
        # 2. 传入数据到 Viewer
        viewer = FeatureViewerApp(planes, cyls, mask, mesh)
        viewer.run()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"提取或渲染过程中发生错误: {e}")