import enum

import numpy as np
import scipy.interpolate


class LASClassificationCode(enum.IntEnum):
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


def normalize_cloud_height(las, *, interpolation_method="nearest"):
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


def create_regular_grid(xyzs, ncols, dx, dy, add_noise=True):
    grid, indices = [], []

    for i, xyz in enumerate(xyzs):
        means = xyz.mean(axis=0, keepdims=True)
        means[0][-1] = 0  # Don't recenter Z
        x = i % ncols * dx + np.random.normal(loc=0.0, scale=1.0) * add_noise
        y = i // ncols * dy + np.random.normal(loc=0.0, scale=1.0) * add_noise
        grid.append(xyz - means + np.array([[x, y, 0]]))
        indices.append(np.zeros(xyz.shape[0], dtype=np.uint32) + i)

    return np.vstack(grid), np.hstack(indices)
