"""Functions for manipulating point clouds."""

import enum
from typing import Optional

import laspy
import numpy as np
import scipy.interpolate


class LASClassificationCode(enum.IntEnum):
    """Class names used in LAS files."""

    NEVER_CLASSIFIED = 0
    UNASSIGNED = 1
    GROUND = 2
    LOW_VEGETATION = 3
    MEDIUM_VEGETATION = 4
    HIGH_VEGETATION = 5
    BUILDING = 6
    LOW_POINT = 7
    WATER = 9
    RAIL = 10
    ROAD_SURFACE = 11
    WIRE_GUARD = 13
    WIRE_CONDUCTOR = 14
    TRANSMISSION_TOWER = 15
    WIRE_STRUCTURE_CONNECTOR = 16
    BRIDGE_DECK = 17
    HIGH_NOISE = 18


def normalize_cloud_height(
    las: laspy.LasData,
    *,
    interpolation_method: str = "nearest",
):
    """Subtract the ground level from all points in a LasData object."""
    assert np.any(las.classification == LASClassificationCode.GROUND)
    out = las.xyz.copy()
    ground_level = scipy.interpolate.griddata(
        points=las.xyz[las.classification == LASClassificationCode.GROUND, :2],
        values=las.xyz[las.classification == LASClassificationCode.GROUND, 2],
        xi=las.xyz[:, :2],
        method=interpolation_method,
    )
    out[:, 2] -= ground_level
    return np.clip(out, a_min=0, a_max=np.inf)


def create_regular_grid(
    xyzs: list[np.ndarray],
    ncols: int,
    dx: float,
    dy: float,
    add_noise: bool = True,
):
    """Arrange a collection of point clouds into a single cloud in a regular grid."""
    grid, indices = [], []

    for i, xyz in enumerate(xyzs):
        means = xyz.mean(axis=0, keepdims=True)
        means[0][-1] = 0  # Don't recenter Z
        x = i % ncols * dx + np.random.normal(loc=0.0, scale=1.0) * add_noise
        y = i // ncols * dy + np.random.normal(loc=0.0, scale=1.0) * add_noise
        grid.append(xyz - means + np.array([[x, y, 0]]))
        indices.append(np.zeros(xyz.shape[0], dtype=np.uint32) + i)

    return np.vstack(grid), np.hstack(indices)


def numpy_to_las(
    xyz: np.ndarray,
    *,
    color: Optional[np.ndarray] = None,
    scale: float = 0.0001,
) -> laspy.LasData:
    """Convert a Numpy array of points into a LasData object.

    Args:
        xyz: An array of point coordinates with shape (N, 3).
        color: An array of colors for every point, shape (N, 3), in range [0, 255].
        scale: Scale used to store the coordinates.

    """

    points = laspy.ScaleAwarePointRecord.zeros(
        xyz.shape[0],
        point_format=laspy.PointFormat(3),
        scales=[scale] * 3,
        offsets=[0] * 3,
    )
    points.x[:] = xyz[:, 0]
    points.y[:] = xyz[:, 1]
    points.z[:] = xyz[:, 2]
    if color is not None:
        points.red[:] = color[:, 0]
        points.green[:] = color[:, 1]
        points.blue[:] = color[:, 2]
    return laspy.LasData(
        header=laspy.LasHeader(),
        points=points,
    )
