from __future__ import annotations
import open3d as o3d
from .logging_utils import log

def load_stl_mesh(path: str) -> o3d.geometry.TriangleMesh:
    log(f"Loading STL mesh: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load STL: {path}")
    mesh.compute_vertex_normals()
    return mesh

def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, n_points: int = 200000) -> o3d.geometry.PointCloud:
    # Uniformly sample points on mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return pcd
