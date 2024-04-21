"""Utility functions."""

import numpy as np


def crop_by_other(points: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Crop points by the extent of other."""
    crop_indices = np.nonzero(
        (points[:, 0] >= other[:, 0].min())
        & (points[:, 1] >= other[:, 1].min())
        & (points[:, 0] <= other[:, 0].max())
        & (points[:, 1] <= other[:, 1].max())
    )
    return points[crop_indices]
