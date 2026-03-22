from __future__ import annotations

import copy
import numpy as np
import open3d as o3d
from .logging_utils import log

#给两个open 3d网格和齐次变换展示配准效果

def show_alignment_mesh(
    cad_mesh: o3d.geometry.TriangleMesh,
    scan_mesh: o3d.geometry.TriangleMesh,
    T_cad_to_scan: np.ndarray,
    show_frame: bool = True,
):
    cad = copy.deepcopy(cad_mesh)
    scan = copy.deepcopy(scan_mesh)

    cad.transform(T_cad_to_scan)

    for mesh in (cad, scan):
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

    cad.paint_uniform_color([1.0, 0.2, 0.2])
    scan.paint_uniform_color([0.75, 0.75, 0.75])

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name="Registration Result (Mesh)",
        width=1280,
        height=800,
        visible=True,
    )
    if not ok:
        log("Visualizer.create_window() failed. (Possible causes: remote/headless session, OpenGL issue)")
        return

    vis.add_geometry(scan)
    vis.add_geometry(cad)

    if show_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])
        vis.add_geometry(frame)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True

    log("Showing mesh registration result window. Close it to continue/exit.")
    vis.run()
    vis.destroy_window()
