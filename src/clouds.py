import enum

import numpy as np
import scipy.interpolate


class LASClassCode(enum.IntEnum):
    NEVER_CLASSIFIED = 0
    UNASSIGNED = 1
    GROUND = 2


def normalize_cloud_height(las):
    out = las.xyz.copy()
    assert np.any(las.classification == LASClassCode.GROUND)
    ground_level = scipy.interpolate.griddata(
        points=las.xyz[las.classification == LASClassCode.GROUND, :2],
        values=las.xyz[las.classification == LASClassCode.GROUND, 2],
        xi=las.xyz[:, :2],
        method="nearest",
    )
    out[:, 2] -= ground_level
    return out
