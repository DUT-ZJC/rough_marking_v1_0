from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import open3d as o3d

from .logging_utils import log

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    from OCC.Core.GProp import GProp_GProps
except Exception as e:  # pragma: no cover
    STEPControl_Reader = None
    _IMPORT_ERR = e


# ===== mm thresholds =====
PLANE_MERGE_ANGLE_DEG = 0.10
PLANE_MERGE_OFFSET_MM = 0.05
MIN_PLANE_AREA_MM2 = 100.0

CYL_MERGE_ANGLE_DEG = 0.10
CYL_MERGE_AXIS_DIST_MM = 0.10
CYL_MERGE_RADIUS_MM = 0.05
MIN_CYL_AREA_MM2 = 50.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _cos_deg(deg: float) -> float:
    return float(np.cos(np.deg2rad(deg)))


def _canon_dir(v: np.ndarray) -> np.ndarray:
    v = _unit(v.astype(np.float64))
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v = -v
    return v


def _face_area(face) -> float:
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    return float(props.Mass())  # mm^2 if model is mm


def _plane_from_occ(geom_plane: Geom_Plane) -> Tuple[np.ndarray, float, np.ndarray]:
    ax3 = geom_plane.Position()
    n = np.array([ax3.Direction().X(), ax3.Direction().Y(), ax3.Direction().Z()], dtype=np.float64)
    n = _canon_dir(n)
    p = geom_plane.Location()
    p0 = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
    d = -float(n @ p0)
    return n, d, p0


def _cyl_from_occ(geom_cyl: Geom_CylindricalSurface) -> Tuple[np.ndarray, np.ndarray, float]:
    ax3 = geom_cyl.Position()
    v = np.array([ax3.Direction().X(), ax3.Direction().Y(), ax3.Direction().Z()], dtype=np.float64)
    v = _canon_dir(v)
    o_gp = ax3.Location()
    o = np.array([o_gp.X(), o_gp.Y(), o_gp.Z()], dtype=np.float64)
    r = float(geom_cyl.Radius())
    return o, v, r


def _axis_line_distance(o1: np.ndarray, v1: np.ndarray, o2: np.ndarray, v2: np.ndarray) -> float:
    v1 = _unit(v1)
    v2 = _unit(v2)
    w0 = o1 - o2
    c = np.cross(v1, v2)
    cn = np.linalg.norm(c)
    if cn < 1e-10:
        return float(np.linalg.norm(np.cross(w0, v1)))
    return float(abs(w0 @ c) / (cn + 1e-12))


def _read_step_shape(step_path: str):
    if STEPControl_Reader is None:
        raise ImportError(f"pythonocc-core required. Import error: {_IMPORT_ERR}")
    log(f"Loading STEP: {step_path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise ValueError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def _face_triangulation_mesh(face) -> Optional[o3d.geometry.TriangleMesh]:
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face, loc)
    if tri is None or tri.NbNodes() == 0 or tri.NbTriangles() == 0:
        return None

    trsf = loc.Transformation()

    V = np.empty((tri.NbNodes(), 3), dtype=np.float64)
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i)
        p.Transform(trsf)
        V[i - 1, :] = (p.X(), p.Y(), p.Z())

    F = np.empty((tri.NbTriangles(), 3), dtype=np.int32)
    for i in range(1, tri.NbTriangles() + 1):
        t = tri.Triangle(i)
        i1, i2, i3 = t.Get()
        F[i - 1, :] = (i1 - 1, i2 - 1, i3 - 1)

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V),
        o3d.utility.Vector3iVector(F),
    )
    m.compute_vertex_normals()
    return m


@dataclass
class CadPlaneFeature:
    id: int
    normal: np.ndarray
    d: float
    p0: np.ndarray
    area_mm2: float
    face_count: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    mesh: o3d.geometry.TriangleMesh
    tri_indices: np.ndarray


@dataclass
class CadCylinderFeature:
    id: int
    axis_origin: np.ndarray
    axis_dir: np.ndarray
    radius: float
    area_mm2: float
    face_count: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    mesh: o3d.geometry.TriangleMesh
    tri_indices: np.ndarray


def extract_cad_planes_and_cylinders(
    step_path: str,
    linear_deflection: float = 0.5
) -> Tuple[
    List[CadPlaneFeature],
    List[CadCylinderFeature],
    o3d.geometry.TriangleMesh,
    o3d.geometry.TriangleMesh,
]:
    """
    Returns:
      planes, cylinders, unknown_mesh, cad_base_mesh_from_faces

    unknown_mesh:
      merged mesh of faces that are NOT analytic plane/cylinder.

    cad_base_mesh_from_faces:
      a CAD base mesh built in the SAME face traversal / triangulation order
      as tri_indices, so feature.tri_indices and cad_base_mesh_from_faces.triangles
      share the same indexing system.
    """
    shape = _read_step_shape(step_path)

    log(f"Tessellating STEP (linear_deflection={linear_deflection})")
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, 0.5, True)

    raw_planes = []
    raw_cyls = []
    unknown_mesh = o3d.geometry.TriangleMesh()

    # 与 tri_indices 同源的 CAD base mesh
    cad_base_mesh_from_faces = o3d.geometry.TriangleMesh()
    global_tri_offset = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        surf = BRep_Tool.Surface(face)

        m = _face_triangulation_mesh(face)
        if m is None:
            exp.Next()
            continue

        # 当前 face 在全局 CAD mesh 中的 triangle 索引范围
        n_face_tri = len(np.asarray(m.triangles))
        face_tri_indices = np.arange(
            global_tri_offset,
            global_tri_offset + n_face_tri,
            dtype=np.int32,
        )

        # 拼入全局 CAD base mesh
        cad_base_mesh_from_faces += m
        global_tri_offset += n_face_tri

        V = np.asarray(m.vertices)
        bmin = V.min(axis=0)
        bmax = V.max(axis=0)
        area = _face_area(face)

        try:
            stype = surf.DynamicType().Name()

            if stype == "Geom_Plane":
                gp = Geom_Plane.DownCast(surf)
                n, d, p0 = _plane_from_occ(gp)
                raw_planes.append({
                    "n": n,
                    "d": float(d),
                    "p0": p0,
                    "area": area,
                    "mesh": m,
                    "bmin": bmin,
                    "bmax": bmax,
                    "tri_indices": face_tri_indices,
                })

            elif stype == "Geom_CylindricalSurface":
                gc = Geom_CylindricalSurface.DownCast(surf)
                o, v, r = _cyl_from_occ(gc)
                raw_cyls.append({
                    "o": o,
                    "v": v,
                    "r": float(r),
                    "area": area,
                    "mesh": m,
                    "bmin": bmin,
                    "bmax": bmax,
                    "tri_indices": face_tri_indices,
                })

            else:
                unknown_mesh += m

        except Exception:
            unknown_mesh += m

        exp.Next()

    log(f"Raw analytic: planes={len(raw_planes)} cylinders={len(raw_cyls)}")

    # ---- merge planes ----
    planes: List[CadPlaneFeature] = []
    used = [False] * len(raw_planes)
    cos_th = _cos_deg(PLANE_MERGE_ANGLE_DEG)
    pid = 0

    for i, pi in enumerate(raw_planes):
        if used[i]:
            continue

        used[i] = True
        n0, d0 = pi["n"], float(pi["d"])

        group_mesh = o3d.geometry.TriangleMesh()
        group_mesh += pi["mesh"]

        area_sum = float(pi["area"])
        bmin = pi["bmin"].copy()
        bmax = pi["bmax"].copy()
        count = 1

        group_tri_indices = [np.asarray(pi["tri_indices"], dtype=np.int32).reshape(-1)]

        for j in range(i + 1, len(raw_planes)):
            if used[j]:
                continue

            pj = raw_planes[j]
            nj, dj = pj["n"], float(pj["d"])

            if abs(float(nj @ n0)) < cos_th:
                continue
            if abs(dj - d0) > PLANE_MERGE_OFFSET_MM:
                continue

            used[j] = True

            group_mesh += pj["mesh"]
            area_sum += float(pj["area"])
            bmin = np.minimum(bmin, pj["bmin"])
            bmax = np.maximum(bmax, pj["bmax"])
            count += 1

            group_tri_indices.append(
                np.asarray(pj["tri_indices"], dtype=np.int32).reshape(-1)
            )

        if area_sum < MIN_PLANE_AREA_MM2:
            continue

        group_mesh.remove_duplicated_vertices()
        group_mesh.remove_duplicated_triangles()
        group_mesh.compute_vertex_normals()

        merged_tri_indices = np.unique(
            np.concatenate(group_tri_indices)
        ).astype(np.int32)

        planes.append(CadPlaneFeature(
            id=pid,
            normal=n0,
            d=d0,
            p0=pi["p0"],
            area_mm2=area_sum,
            face_count=count,
            bbox_min=bmin,
            bbox_max=bmax,
            mesh=group_mesh,
            tri_indices=merged_tri_indices,
        ))
        pid += 1

    # ---- merge cylinders ----
    cyls: List[CadCylinderFeature] = []
    usedc = [False] * len(raw_cyls)
    cos_cyl = _cos_deg(CYL_MERGE_ANGLE_DEG)
    cid = 0

    for i, ci in enumerate(raw_cyls):
        if usedc[i]:
            continue

        usedc[i] = True
        o0, v0, r0 = ci["o"], ci["v"], float(ci["r"])

        group_mesh = o3d.geometry.TriangleMesh()
        group_mesh += ci["mesh"]

        area_sum = float(ci["area"])
        bmin = ci["bmin"].copy()
        bmax = ci["bmax"].copy()
        count = 1

        group_tri_indices = [np.asarray(ci["tri_indices"], dtype=np.int32).reshape(-1)]

        for j in range(i + 1, len(raw_cyls)):
            if usedc[j]:
                continue

            cj = raw_cyls[j]
            oj, vj, rj = cj["o"], cj["v"], float(cj["r"])

            if abs(float(vj @ v0)) < cos_cyl:
                continue
            if abs(rj - r0) > CYL_MERGE_RADIUS_MM:
                continue
            if _axis_line_distance(o0, v0, oj, vj) > CYL_MERGE_AXIS_DIST_MM:
                continue

            usedc[j] = True

            group_mesh += cj["mesh"]
            area_sum += float(cj["area"])
            bmin = np.minimum(bmin, cj["bmin"])
            bmax = np.maximum(bmax, cj["bmax"])
            count += 1

            group_tri_indices.append(
                np.asarray(cj["tri_indices"], dtype=np.int32).reshape(-1)
            )

        if area_sum < MIN_CYL_AREA_MM2:
            continue

        group_mesh.remove_duplicated_vertices()
        group_mesh.remove_duplicated_triangles()
        group_mesh.compute_vertex_normals()

        merged_tri_indices = np.unique(
            np.concatenate(group_tri_indices)
        ).astype(np.int32)

        cyls.append(CadCylinderFeature(
            id=cid,
            axis_origin=o0,
            axis_dir=v0,
            radius=r0,
            area_mm2=area_sum,
            face_count=count,
            bbox_min=bmin,
            bbox_max=bmax,
            mesh=group_mesh,
            tri_indices=merged_tri_indices,
        ))
        cid += 1

    # finalize unknown mesh
    if len(unknown_mesh.vertices) > 0 and len(unknown_mesh.triangles) > 0:
        unknown_mesh.remove_duplicated_vertices()
        unknown_mesh.remove_duplicated_triangles()
        unknown_mesh.compute_vertex_normals()

    # 注意：这里不要去重，否则会破坏 tri_indices 与 triangles 的对应关系
    if len(cad_base_mesh_from_faces.vertices) > 0 and len(cad_base_mesh_from_faces.triangles) > 0:
        cad_base_mesh_from_faces.compute_vertex_normals()

    log(
        f"Merged features: planes={len(planes)} cylinders={len(cyls)} "
        f"unknown_mesh_tris={len(unknown_mesh.triangles)} "
        f"cad_base_mesh_tris={len(cad_base_mesh_from_faces.triangles)}"
    )

    return planes, cyls, unknown_mesh, cad_base_mesh_from_faces
