from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from .logging_utils import log

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopLoc import TopLoc_Location
except Exception as e:  # pragma: no cover
    STEPControl_Reader = None
    _IMPORT_ERR = e


@dataclass
class FaceMesh:
    face_id: int
    mesh: o3d.geometry.TriangleMesh


def _read_step_shape(step_path: str):
    """Read STEP -> TopoDS_Shape."""
    if STEPControl_Reader is None:
        raise ImportError(
            "pythonocc-core is required to read STEP in this project. "
            f"Import error: {_IMPORT_ERR}"
        )
    log(f"Loading STEP: {step_path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise ValueError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def load_step_faces_as_o3d_meshes(step_path: str, linear_deflection: float = 0.5) -> list[FaceMesh]:
    """
    Tessellate STEP and return per-face Open3D triangle meshes for CAD face picking.

    NOTE: This function uses pythonocc-core 7.9.x API:
      - triangulation.Node(i)
      - triangulation.Triangle(i)
    """
    shape = _read_step_shape(step_path)

    # Tessellate whole shape once, so each face has triangulation
    log(f"Tessellating STEP (linear_deflection={linear_deflection})")
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, 0.5, True)

    face_meshes: list[FaceMesh] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)

    fid = 0
    while exp.More():
        face = topods.Face(exp.Current())

        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            exp.Next()
            continue

        nb_nodes = tri.NbNodes()
        nb_tris = tri.NbTriangles()
        if nb_nodes == 0 or nb_tris == 0:
            exp.Next()
            continue

        # Apply face location transformation
        trsf = loc.Transformation()

        # vertices
        V = np.empty((nb_nodes, 3), dtype=np.float64)
        for i in range(1, nb_nodes + 1):
            p = tri.Node(i)          # gp_Pnt
            p.Transform(trsf)        # map to global
            V[i - 1, :] = (p.X(), p.Y(), p.Z())

        # triangles
        F = np.empty((nb_tris, 3), dtype=np.int32)
        for i in range(1, nb_tris + 1):
            t = tri.Triangle(i)
            i1, i2, i3 = t.Get()
            F[i - 1, :] = (i1 - 1, i2 - 1, i3 - 1)

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V),
            o3d.utility.Vector3iVector(F)
        )
        mesh.compute_vertex_normals()

        face_meshes.append(FaceMesh(face_id=fid, mesh=mesh))
        fid += 1

        exp.Next()

    if not face_meshes:
        raise ValueError(
            "No face meshes produced from STEP. "
            "Try smaller linear_deflection (e.g. 0.2) or check STEP content."
        )

    log(f"STEP faces tessellated: {len(face_meshes)} faces")
    return face_meshes
