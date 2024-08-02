import enum

import numpy as np
import scipy.interpolate


class LASClassCode(enum.IntEnum):
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
