"""Python API for lidar_pro CUDA extension."""

from ._C import (
    points2img,
    points2img2,
    points2rgb,
    trilinear_interpolation_bw,
    trilinear_interpolation_fw,
)


def lidar_pro(points, camera_in, depth=80.0, radius1=1, ww=1241, hh=376):
    """Unified depth projection API."""
    return points2img2(points, camera_in, depth, radius1, ww, hh)


lidar_pro_points2img = points2img
lidar_pro_points2img2 = points2img2
lidar_pro_points2rgb = points2rgb


__all__ = [
    "lidar_pro",
    "lidar_pro_points2img",
    "lidar_pro_points2img2",
    "lidar_pro_points2rgb",
    "points2img",
    "points2img2",
    "points2rgb",
    "trilinear_interpolation_fw",
    "trilinear_interpolation_bw",
]
