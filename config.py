# ====== User editable paths ======
CAD_STEP_PATH = r"./data/2_orginal.step"  # STEP file of ideal model
SCAN_STL_PATH = r"./data/2.stl"           # STL triangle mesh of scanned rough part
SCAN_FEATURE_BACKEND = "step_guided"      # "stl" | "step_guided"

# ====== Processing params ======
VOXEL_SIZE = 2.0            # mm-like unit if your data is in mm
NORMAL_RADIUS = 6.0         # for normal estimation
FEATURE_PLANE_THRESH = 1.5  # RANSAC distance threshold for plane (same unit as data)
MAX_PLANES = 6              # detect top-K planes

# Registration params
RANSAC_DIST = 4.0
ICP_DIST = 3.0
MAX_CORR = 50000
COARSE_ENCLOSE_ENABLE = True
COARSE_ENCLOSE_MARGIN = 0.0
COARSE_ENCLOSE_SAMPLE_POINTS = 12000
COARSE_ENCLOSE_MAX_ITERS = 8

# STEP-guided scan feature fitting
STEP_GUIDED_LINEAR_DEFLECTION = 0.5
STEP_GUIDED_SUPPORT_GAP_MM = 2.0
STEP_GUIDED_PLANE_TOL_MM = 1.0
STEP_GUIDED_CYLINDER_TOL_MM = 1.0
STEP_GUIDED_GENERIC_TOL_MM = 1.5
STEP_GUIDED_MIN_SUPPORT_TRIANGLES = 20
STEP_GUIDED_USE_GLOBAL_RESULTS = True

# Constrained optimizer weights (smaller tolerance => larger weight)
DATUM_PLANE_ANGLE_TOL_DEG = 0.10
DATUM_PLANE_OFFSET_TOL = 0.10
DATUM_WEIGHT = 1.0

# Visualization
SHOW_DEBUG_FEATURES = True
