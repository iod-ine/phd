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
    las_list: list[laspy.LasData],
    ncols: int,
    dx: float,
    dy: float,
    xy_noise_mean: float = 0.0,
    xy_noise_std: float = 0.0,
    features_to_extract: Optional[list[str]] = None,
    height_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Arrange a collection of point clouds into a single cloud in a regular grid.

    Args:
        las_list: List of LAS objects to combine.
        ncols: Number of columns in the grid.
        dx: Step in the X direction.
        dy: Step in the Y direction.
        xy_noise_mean: Mean of the normal distribution of XY noise.
        xy_noise_std: Standard deviation of the normal distribution of XY noise.
        features_to_extract: List of features to extract (have to be dimensions of the
            LAS files).
        height_threshold: Points lower then this threshold are filtered out.

    Returns:
        pos, x, y: Coordinates, features, labels (indices of the objects in las_list).
    """
    coords, features, indices = [], [], []
    rng = np.random.default_rng()

    for i, las in enumerate(las_list):
        xyz = las.xyz[las.xyz[:, 2] >= height_threshold]
        means = xyz.mean(axis=0, keepdims=True)
        means[0][-1] = 0  # Don't recenter Z
        x = i % ncols * dx + rng.normal(loc=xy_noise_mean, scale=xy_noise_std)
        y = i // ncols * dy + rng.normal(loc=xy_noise_mean, scale=xy_noise_std)
        coords.append(xyz - means + np.array([[x, y, 0]]))
        features.append(extract_las_features(las, features_to_extract))
        indices.append(np.zeros(xyz.shape[0], dtype=np.uint32) + i)

    return np.vstack(coords), np.vstack(features), np.hstack(indices)


def numpy_to_las(
    xyz: np.ndarray,
    *,
    color: Optional[np.ndarray] = None,
    scale: float = 0.0001,
    extra_dim: Optional[np.ndarray] = None,
) -> laspy.LasData:
    """Convert a Numpy array of points into a LasData object.

    Args:
        xyz: An array of point coordinates with shape (N, 3).
        color: An array of colors for every point, shape (N, 3), in range [0, 255].
        scale: Scale used to store the coordinates.
        extra_dim: An extra dimension to add to the points, shape (N,).

    Notes:
        https://laspy.readthedocs.io/en/latest/intro.html

    """
    points = laspy.ScaleAwarePointRecord.zeros(
        xyz.shape[0],
        point_format=laspy.PointFormat(3),
        scales=[scale] * 3,
        offsets=np.min(xyz, axis=0),
    )
    points.x[:] = xyz[:, 0]
    points.y[:] = xyz[:, 1]
    points.z[:] = xyz[:, 2]
    if color is not None:
        points.red[:] = color[:, 0]
        points.green[:] = color[:, 1]
        points.blue[:] = color[:, 2]

    header = laspy.LasHeader(point_format=laspy.PointFormat(3))
    header.scales = points.scales
    header.point_count = xyz.shape[0]

    las = laspy.LasData(
        header=header,
        points=points,
    )

    if extra_dim is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(name="extra", type=extra_dim.dtype))
        las.extra = extra_dim

    return las


def extract_las_features(
    las: laspy.LasData,
    features_to_extract: Optional[list[str]] = None,
) -> np.ndarray:
    """Extract a set of features from the LAS file.

    Args:
        las: The LAS to extract features from.
        features_to_extract: List of features to extract (have to be dimensions of the
            LAS file).

    Returns:
        features: An array of features shaped (N, num_features).

    Notes:
        See https://laspy.readthedocs.io/en/latest/intro.html#point-records for possible
        dimensions that can be extracted depending on the point format of the file. The
        default set includes Intensity, Return number, Number of returns, and
        Classification, in that order.

    """
    features_to_extract = features_to_extract or [
        "intensity",
        "return_number",
        "number_of_returns",
        "classification",
    ]
    features = np.empty(
        shape=(las.header.point_count, len(features_to_extract)),
        dtype=np.float32,
    )
    for i, feature in enumerate(features_to_extract):
        features[:, i] = las[feature]
    return features
