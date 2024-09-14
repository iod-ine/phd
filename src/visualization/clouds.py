"""Functions for visualizing point clouds."""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np


def scatter_point_cloud_2d(
    xyz: np.ndarray,
    projection: Literal["XZ", "YZ", "XY"],
    *,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Create a 2D scatter plot of a point cloud."""
    if ax is None:
        fig, ax = plt.subplots()
    match projection:
        case "XZ":
            x, y = xyz[:, 0], xyz[:, 2]
        case "YZ":
            x, y = xyz[:, 1], xyz[:, 2]
        case "XY":
            x, y = xyz[:, 0], xyz[:, 1]
    ax.scatter(x, y, **kwargs)
    return ax


def scatter_point_cloud_3d(
    xyz: np.ndarray,
    *,
    ax=None,
    color=None,
    figsize=None,
    elev=None,
    azim=None,
    cmap=None,
) -> plt.Axes:
    """Create a 3D scatter plot of a point cloud."""
    color = color if color is not None else xyz[:, 2]
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(*np.rollaxis(xyz, 1), c=color, s=1, cmap=cmap)
    ax.set_aspect("equal")
    return ax
