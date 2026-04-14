"""Standalone entry point for the STEP-guided STL fitting viewer."""

from __future__ import annotations

import argparse
import os

from .core_types import FitThresholds, RegistrationConfig
from .pipeline import StepSTLFitSession
from .view import StepFitViewerApp


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the demo viewer."""
    parser = argparse.ArgumentParser(
        description="STEP 解析面引导的 STL 局部拟合与毛刺标注演示程序。",
    )
    parser.add_argument("--step", type=str, default=r"./data/2_orginal.step", help="STEP 文件路径。")
    parser.add_argument("--stl", type=str, default=r"./data/2.stl", help="STL 文件路径。")
    parser.add_argument(
        "--linear-deflection",
        type=float,
        default=0.5,
        help="STEP 面三角化线性误差，越小越精细。",
    )
    parser.add_argument("--support-gap", type=float, default=2.0, help="局部支撑搜索距离，单位 mm。")
    parser.add_argument("--plane-tol", type=float, default=1.0, help="平面拟合残差阈值，单位 mm。")
    parser.add_argument("--cylinder-tol", type=float, default=1.0, help="柱面拟合残差阈值，单位 mm。")
    parser.add_argument("--generic-tol", type=float, default=1.5, help="锥/球/环面拟合残差阈值，单位 mm。")
    parser.add_argument(
        "--min-support-triangles",
        type=int,
        default=20,
        help="局部支撑区域允许的最小三角片数量。",
    )
    parser.add_argument("--voxel-size", type=float, default=2.0, help="粗配准体素尺寸。")
    parser.add_argument("--normal-radius", type=float, default=6.0, help="粗配准法向搜索半径。")
    return parser


def main() -> None:
    """Parse arguments, build the fitting session, and start the viewer."""
    args = build_parser().parse_args()
    if not os.path.exists(args.step):
        raise FileNotFoundError(f"STEP not found: {args.step}")
    if not os.path.exists(args.stl):
        raise FileNotFoundError(f"STL not found: {args.stl}")

    session = StepSTLFitSession(
        step_path=args.step,
        scan_stl_path=args.stl,
        linear_deflection=float(args.linear_deflection),
        registration_config=RegistrationConfig(
            VOXEL_SIZE=float(args.voxel_size),
            NORMAL_RADIUS=float(args.normal_radius),
        ),
    )
    session.load()

    viewer = StepFitViewerApp(
        session=session,
        thresholds=FitThresholds(
            support_gap_mm=float(args.support_gap),
            plane_tol_mm=float(args.plane_tol),
            cylinder_tol_mm=float(args.cylinder_tol),
            generic_tol_mm=float(args.generic_tol),
            min_support_triangles=int(args.min_support_triangles),
        ),
    )
    viewer.run()


if __name__ == "__main__":
    main()
