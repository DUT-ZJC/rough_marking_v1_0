from __future__ import annotations
import time
import numpy as np

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def validate_feature_triangles(base_mesh, features, name: str) -> None:
    n_tri = len(np.asarray(base_mesh.triangles))
    for feat in features:
        tri_ids = np.asarray(getattr(feat, "tri_indices", []), dtype=np.int32).reshape(-1)
        if tri_ids.size == 0:
            print(f"[WARN] {name}#{feat.id} has empty tri_indices")
            continue
        if tri_ids.min() < 0 or tri_ids.max() >= n_tri:
            print(
                f"[ERROR] {name}#{feat.id} tri_indices out of range: "
                f"min={tri_ids.min()} max={tri_ids.max()} n_tri={n_tri}"
            )
#特征检查工具