from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .logging_utils import log
from .feature_detect import PlaneFeature


# -----------------------------
# math helpers
# -----------------------------
def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy],
                     [wz, 0.0, -wx],
                     [-wy, wx, 0.0]], dtype=np.float64)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map for small se3: xi=[w(3), v(3)]"""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w = xi[:3]
    v = xi[3:]
    th = float(np.linalg.norm(w))

    if th < 1e-12:
        R = np.eye(3) + skew(w)
        V = np.eye(3) + 0.5 * skew(w)
    else:
        wn = w / th
        K = skew(wn)
        R = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
        V = np.eye(3) + (1.0 - np.cos(th)) / th * K + (th - np.sin(th)) / th * (K @ K)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def transform_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ X.T).T + T[:3, 3]


def _finite(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        idx = tuple(bad[0].tolist()) if bad.size else ()
        raise ValueError(f"{name} contains NaN/Inf at {idx}")


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)


# -----------------------------
# SAFE NN: voxel hash (pure numpy)
# -----------------------------
def build_voxel_hash(points: np.ndarray, voxel: float):
    pts = np.asarray(points, dtype=np.float64)
    voxel = float(voxel)
    origin = pts.min(axis=0) - voxel * 0.5
    ijk = np.floor((pts - origin) / voxel).astype(np.int32)

    table: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(map(tuple, ijk)):
        table.setdefault(key, []).append(i)

    table2: dict[tuple[int, int, int], np.ndarray] = {}
    for k, lst in table.items():
        table2[k] = np.asarray(lst, dtype=np.int32)
    return voxel, origin, table2


def nn_search_voxel_hash(X: np.ndarray,
                         target_pts: np.ndarray,
                         voxel: float,
                         origin: np.ndarray,
                         table: dict,
                         max_dist: float):
    X = np.asarray(X, dtype=np.float64)
    max_dist = float(max_dist)
    max_dist2 = max_dist * max_dist

    nn_idx = np.full((X.shape[0],), -1, dtype=np.int64)
    nn_d2 = np.full((X.shape[0],), np.inf, dtype=np.float64)

    ijk = np.floor((X - origin) / voxel).astype(np.int32)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for i in range(X.shape[0]):
        key0 = ijk[i]
        best_j = -1
        best_d2 = np.inf

        for dx, dy, dz in offs:
            key = (int(key0[0] + dx), int(key0[1] + dy), int(key0[2] + dz))
            cand = table.get(key, None)
            if cand is None or cand.size == 0:
                continue
            P = target_pts[cand]  # (M,3)
            d2 = np.sum((P - X[i]) ** 2, axis=1)
            jloc = int(np.argmin(d2))
            d2min = float(d2[jloc])
            if d2min < best_d2:
                best_d2 = d2min
                best_j = int(cand[jloc])

        if best_j >= 0 and best_d2 < max_dist2:
            nn_idx[i] = best_j
            nn_d2[i] = best_d2

    nn_dist = np.sqrt(nn_d2, where=np.isfinite(nn_d2), out=np.full_like(nn_d2, np.inf))
    return nn_idx, nn_dist


# -----------------------------
# pure 6x6 solver (avoid np.linalg.solve / MKL)
# -----------------------------
def solve_6x6(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b for 6x6 A using Gaussian elimination with partial pivoting.
    Pure numpy operations, no BLAS/LAPACK calls.
    """
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).reshape(6).copy()

    # augment
    M = np.zeros((6, 7), dtype=np.float64)
    M[:, :6] = A
    M[:, 6] = b

    for col in range(6):
        # pivot
        pivot = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[pivot, col]) < 1e-14:
            raise np.linalg.LinAlgError("Singular matrix in solve_6x6")
        if pivot != col:
            M[[col, pivot], :] = M[[pivot, col], :]

        # normalize pivot row
        pv = M[col, col]
        M[col, col:] /= pv

        # eliminate
        for row in range(6):
            if row == col:
                continue
            factor = M[row, col]
            if abs(factor) > 0:
                M[row, col:] -= factor * M[col, col:]

    x = M[:, 6]
    return x


# -----------------------------
# data models
# -----------------------------
@dataclass
class CylinderFeature:
    axis_origin: np.ndarray  # (3,)
    axis_dir: np.ndarray     # (3,) unit (or close)
    radius: float = 0.0


@dataclass
class OptResult:
    T: np.ndarray
    rmse: float
    iters: int


# -----------------------------
# optimizer
# -----------------------------
def constrained_point_to_plane_icp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    target_normals: np.ndarray,
    T0: np.ndarray,
    max_corr_dist: float,
    max_iters: int,
    datum_pairs: list[tuple[PlaneFeature, PlaneFeature]] | None = None,
    datum_angle_tol_deg: float = 0.1,
    datum_offset_tol: float = 0.1,
    datum_weight: float = 1.0,
    cylinder_pairs: list[tuple[CylinderFeature, CylinderFeature]] | None = None,
    cyl_angle_tol_deg: float = 0.1,
    cyl_axis_offset_tol: float = 0.1,
    cyl_weight: float = 1.0,
    max_corr: int = 50000
) -> OptResult:
    log("Constrained fine registration (custom GN point-to-plane ICP)")

    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    target_normals = np.asarray(target_normals, dtype=np.float64)

    _finite("source_pts", source_pts)
    _finite("target_pts", target_pts)
    _finite("target_normals", target_normals)

    if source_pts.ndim != 2 or source_pts.shape[1] != 3:
        raise ValueError(f"source_pts shape invalid: {source_pts.shape}")
    if target_pts.ndim != 2 or target_pts.shape[1] != 3:
        raise ValueError(f"target_pts shape invalid: {target_pts.shape}")
    if target_normals.shape != target_pts.shape:
        raise ValueError(f"target_normals shape mismatch: {target_normals.shape} vs {target_pts.shape}")
    if len(source_pts) < 200 or len(target_pts) < 200:
        raise ValueError(f"Too few points: src={len(source_pts)} tgt={len(target_pts)}")

    T = np.asarray(T0, dtype=np.float64).copy()
    if T.shape != (4, 4):
        raise ValueError(f"T0 must be 4x4, got {T.shape}")

    # weights from tolerances (normalize by tol^2)
    w_ang = float(datum_weight) / (np.deg2rad(float(datum_angle_tol_deg)) ** 2 + 1e-12)
    w_off = float(datum_weight) / (float(datum_offset_tol) ** 2 + 1e-12)
    w_cang = float(cyl_weight) / (np.deg2rad(float(cyl_angle_tol_deg)) ** 2 + 1e-12)
    w_coff = float(cyl_weight) / (float(cyl_axis_offset_tol) ** 2 + 1e-12)

    # NN index (voxel hash)
    vox = float(max_corr_dist)
    vox_size, vox_origin, vox_table = build_voxel_hash(target_pts, vox)

    rmse_last = 1e18

    for it in range(int(max_iters)):
        log(f"  iter={it:02d} begin")

        X = transform_points(T, source_pts)

        # downsample correspondences BEFORE NN
        if X.shape[0] > max_corr:
            sel = np.random.choice(X.shape[0], size=max_corr, replace=False)
            Xs = X[sel]
            Ss = source_pts[sel]
        else:
            Xs = X
            Ss = source_pts

        nn_idx, _ = nn_search_voxel_hash(Xs, target_pts, vox_size, vox_origin, vox_table, max_corr_dist)
        valid = nn_idx >= 0
        if int(valid.sum()) < 50:
            log("  Too few correspondences; stop.")
            break

        vidx = np.where(valid)[0]
        Xv = Xs[vidx]
        Sv = Ss[vidx]
        Pv = target_pts[nn_idx[vidx]]
        Nv = target_normals[nn_idx[vidx]]

        R = T[:3, :3]
        t = T[:3, 3]
        RS = (R @ Sv.T).T

        # point-to-plane residual
        r = np.einsum("ij,ij->i", Nv, (Xv - Pv))

        # Jacobian: dr/dw = (R*s) x n ; dr/dv = n
        C = np.cross(RS, Nv)
        J = np.zeros((len(vidx), 6), dtype=np.float64)
        J[:, :3] = C
        J[:, 3:] = Nv

        A = J.T @ J
        b = -J.T @ r

        # ---- plane datum constraints ----
        if datum_pairs:
            for cad_plane, scan_plane in datum_pairs:
                n_s = _unit(scan_plane.normal)
                x0 = np.asarray(scan_plane.centroid, dtype=np.float64).reshape(3)

                n_c0 = _unit(cad_plane.normal)
                d_c0 = float(cad_plane.d)

                n_c = R @ n_c0

                # angle (sign-invariant)
                sign = 1.0 if float(n_c @ n_s) >= 0.0 else -1.0
                n_s_eff = sign * n_s
                ang_res = 1.0 - abs(float(n_c @ n_s))

                g = np.cross(n_c, n_s_eff)
                J_ang = np.zeros((1, 6), dtype=np.float64)
                J_ang[0, :3] = g
                r_ang = np.array([ang_res], dtype=np.float64)

                A += w_ang * (J_ang.T @ J_ang)
                b += -w_ang * (J_ang.T @ r_ang)

                # offset
                d_ct = d_c0 - float(n_c @ t)
                off_res = float(n_c @ x0 + d_ct)

                J_off = np.zeros((1, 6), dtype=np.float64)
                J_off[0, :3] = np.cross(x0, n_c)
                J_off[0, 3:] = -n_c
                r_off = np.array([off_res], dtype=np.float64)

                A += w_off * (J_off.T @ J_off)
                b += -w_off * (J_off.T @ r_off)

        # ---- cylinder constraints ----
        if cylinder_pairs:
            for cad_cyl, scan_cyl in cylinder_pairs:
                v_s = _unit(scan_cyl.axis_dir)
                o_s = np.asarray(scan_cyl.axis_origin, dtype=np.float64).reshape(3)

                v_c0 = _unit(cad_cyl.axis_dir)
                o_c0 = np.asarray(cad_cyl.axis_origin, dtype=np.float64).reshape(3)

                v_c = R @ v_c0
                o_c = R @ o_c0 + t

                # axis angle (sign-invariant)
                sign = 1.0 if float(v_c @ v_s) >= 0.0 else -1.0
                v_s_eff = sign * v_s
                ang_res = 1.0 - abs(float(v_c @ v_s))

                g = np.cross(v_c, v_s_eff)
                J_cang = np.zeros((1, 6), dtype=np.float64)
                J_cang[0, :3] = g
                r_cang = np.array([ang_res], dtype=np.float64)

                A += w_cang * (J_cang.T @ J_cang)
                b += -w_cang * (J_cang.T @ r_cang)

                # coaxial offset: perpendicular component of (o_s - o_c) wrt scan axis
                Pperp = np.eye(3) - np.outer(v_s, v_s)
                e = (o_s - o_c)
                r_vec = Pperp @ e  # (3,)

                Ro = R @ o_c0
                J_off3 = np.zeros((3, 6), dtype=np.float64)
                J_off3[:, :3] = Pperp @ skew(Ro)
                J_off3[:, 3:] = -Pperp

                A += w_coff * (J_off3.T @ J_off3)
                b += -w_coff * (J_off3.T @ r_vec)

        # damping
        A += 1e-6 * np.eye(6)

        # >>> critical: DO NOT call np.linalg.solve (MKL). Use pure solve_6x6
        try:
            xi = solve_6x6(A, b)
        except Exception as e:
            log(f"  solve_6x6 failed: {e}")
            break

        if not np.isfinite(xi).all():
            raise ValueError("xi contains NaN/Inf (diverged)")

        T = se3_exp(xi) @ T

        rmse = float(np.sqrt(np.mean(r ** 2)))
        log(f"  iter={it:02d} rmse={rmse:.4f} corr={len(vidx)} |xi|={np.linalg.norm(xi):.3e}")

        if abs(rmse_last - rmse) < 1e-6 and np.linalg.norm(xi) < 1e-6:
            break
        rmse_last = rmse

    # final rmse (rough)
    Xf = transform_points(T, source_pts)
    if Xf.shape[0] > max_corr:
        sel = np.random.choice(Xf.shape[0], size=max_corr, replace=False)
        Xf = Xf[sel]
    nn_idx, nn_d = nn_search_voxel_hash(Xf, target_pts, vox_size, vox_origin, vox_table, max_corr_dist)
    vv = nn_idx >= 0
    rmse_final = float(np.sqrt(np.mean(nn_d[vv] ** 2))) if int(vv.sum()) > 0 else float("inf")

    return OptResult(T=T, rmse=rmse_final, iters=it + 1)
