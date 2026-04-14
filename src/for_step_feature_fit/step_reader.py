"""STEP face extraction for the STEP-guided STL fitting workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d

from ..logging_utils import log
from .core_types import StepAnalyticFace

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_Torus,
    )
except Exception as e:  # pragma: no cover
    STEPControl_Reader = None
    _IMPORT_ERR = e


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _canon_dir(v: np.ndarray) -> np.ndarray:
    v = _unit(v.astype(np.float64))
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0.0:
        v = -v
    return v


def _read_step_shape(step_path: str):
    """Load the top-level OpenCascade shape from a STEP file."""
    if STEPControl_Reader is None:
        raise ImportError(
            "pythonocc-core is required for STEP-driven analytic fitting. "
            f"Import error: {_IMPORT_ERR}"
        )

    log(f"Loading STEP: {step_path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise ValueError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def _face_mesh(face) -> o3d.geometry.TriangleMesh | None:
    """Convert one OCC face triangulation into an Open3D triangle mesh."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face, loc)
    if tri is None or tri.NbNodes() == 0 or tri.NbTriangles() == 0:
        return None

    trsf = loc.Transformation()
    vertices = np.empty((tri.NbNodes(), 3), dtype=np.float64)
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i)
        p.Transform(trsf)
        vertices[i - 1, :] = (p.X(), p.Y(), p.Z())

    triangles = np.empty((tri.NbTriangles(), 3), dtype=np.int32)
    for i in range(1, tri.NbTriangles() + 1):
        t = tri.Triangle(i)
        i1, i2, i3 = t.Get()
        triangles[i - 1, :] = (i1 - 1, i2 - 1, i3 - 1)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    return mesh


def _plane_params(surface) -> dict[str, Any]:
    """Extract canonical plane parameters from an OCC plane surface."""
    plane = surface.Plane()
    ax3 = plane.Position()
    normal = _canon_dir(np.array([
        ax3.Direction().X(),
        ax3.Direction().Y(),
        ax3.Direction().Z(),
    ], dtype=np.float64))
    p = plane.Location()
    point = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
    return {
        "normal": normal,
        "point": point,
        "d": -float(normal @ point),
    }


def _cylinder_params(surface) -> dict[str, Any]:
    """Extract canonical cylinder parameters from an OCC cylinder surface."""
    cyl = surface.Cylinder()
    ax3 = cyl.Position()
    axis_dir = _canon_dir(np.array([
        ax3.Direction().X(),
        ax3.Direction().Y(),
        ax3.Direction().Z(),
    ], dtype=np.float64))
    p = ax3.Location()
    axis_origin = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
    return {
        "axis_origin": axis_origin,
        "axis_dir": axis_dir,
        "radius": float(cyl.Radius()),
    }


def _cone_params(surface) -> dict[str, Any]:
    """Extract canonical cone parameters from an OCC cone surface."""
    cone = surface.Cone()
    ax3 = cone.Position()
    axis_dir = _canon_dir(np.array([
        ax3.Direction().X(),
        ax3.Direction().Y(),
        ax3.Direction().Z(),
    ], dtype=np.float64))
    apex = cone.Apex()
    return {
        "apex": np.array([apex.X(), apex.Y(), apex.Z()], dtype=np.float64),
        "axis_dir": axis_dir,
        "semi_angle_rad": float(cone.SemiAngle()),
        "ref_radius": float(cone.RefRadius()),
    }


def _sphere_params(surface) -> dict[str, Any]:
    """Extract canonical sphere parameters from an OCC sphere surface."""
    sphere = surface.Sphere()
    c = sphere.Location()
    return {
        "center": np.array([c.X(), c.Y(), c.Z()], dtype=np.float64),
        "radius": float(sphere.Radius()),
    }


def _torus_params(surface) -> dict[str, Any]:
    """Extract canonical torus parameters from an OCC torus surface."""
    torus = surface.Torus()
    ax3 = torus.Position()
    axis_dir = _canon_dir(np.array([
        ax3.Direction().X(),
        ax3.Direction().Y(),
        ax3.Direction().Z(),
    ], dtype=np.float64))
    c = ax3.Location()
    return {
        "center": np.array([c.X(), c.Y(), c.Z()], dtype=np.float64),
        "axis_dir": axis_dir,
        "major_radius": float(torus.MajorRadius()),
        "minor_radius": float(torus.MinorRadius()),
    }


def _extract_surface_type_and_params(face) -> tuple[str | None, dict[str, Any]]:
    """Map an OCC face to a supported analytic surface type and parameters."""
    surface = BRepAdaptor_Surface(face, True)
    surface_type = surface.GetType()
    if surface_type == GeomAbs_Plane:
        return "plane", _plane_params(surface)
    if surface_type == GeomAbs_Cylinder:
        return "cylinder", _cylinder_params(surface)
    if surface_type == GeomAbs_Cone:
        return "cone", _cone_params(surface)
    if surface_type == GeomAbs_Sphere:
        return "sphere", _sphere_params(surface)
    if surface_type == GeomAbs_Torus:
        return "torus", _torus_params(surface)
    return None, {}


def extract_step_analytic_faces(
    step_path: str,
    linear_deflection: float = 0.5,
) -> tuple[list[StepAnalyticFace], o3d.geometry.TriangleMesh]:
    """Extract all supported analytic faces from a STEP file.

    Each returned face keeps its own triangulated patch and analytic parameters.
    The merged mesh is used later for coarse global registration against the STL.
    """
    shape = _read_step_shape(step_path)

    log(f"Tessellating STEP (linear_deflection={linear_deflection})")
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, 0.5, True)

    faces: list[StepAnalyticFace] = []
    merged_mesh = o3d.geometry.TriangleMesh()

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()

        mesh = _face_mesh(face)
        if mesh is None or len(mesh.triangles) == 0:
            continue

        surface_type, params = _extract_surface_type_and_params(face)
        if surface_type is None:
            continue

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces.append(
            StepAnalyticFace(
                id=face_id,
                surface_type=surface_type,
                area_mm2=float(mesh.get_surface_area()),
                mesh=mesh,
                params=params,
                bbox_min=vertices.min(axis=0),
                bbox_max=vertices.max(axis=0),
            )
        )
        merged_mesh += mesh
        face_id += 1

    if len(merged_mesh.triangles) > 0:
        merged_mesh.remove_duplicated_vertices()
        merged_mesh.remove_duplicated_triangles()
        merged_mesh.compute_vertex_normals()

    log(f"STEP analytic faces extracted: {len(faces)}")
    return faces, merged_mesh
