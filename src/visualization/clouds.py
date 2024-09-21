"""Functions for visualizing point clouds."""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

import src.clouds


def scatter_point_cloud_2d(
    xyz: np.ndarray,
    projection: Literal["XZ", "YZ", "XY"],
    *,
    ax: Optional[plt.Axes] = None,
    sort_by_height: bool = True,
    recenter: bool = True,
    color=None,
    s=2,
    **kwargs,
) -> plt.Axes:
    """Create a 2D scatter plot of a point cloud."""
    if ax is None:
        fig, ax = plt.subplots()
    if recenter:
        xyz = src.clouds.recenter_cloud(xyz)
    match projection:
        case "XZ":
            if sort_by_height:
                xyz = xyz[np.argsort(xyz[:, 2])]
            x, y = xyz[:, 0], xyz[:, 2]
        case "YZ":
            x, y = xyz[:, 1], xyz[:, 2]
        case "XY":
            x, y = xyz[:, 0], xyz[:, 1]
    color = color if color is not None else xyz[:, 2]
    ax.scatter(x, y, s=s, c=color, **kwargs)
    ax.set_aspect("equal")
    return ax


def scatter_point_cloud_3d(
    xyz: np.ndarray,
    *,
    ax=None,
    recenter: bool = True,
    color=None,
    s=1,
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
    if recenter:
        xyz = src.clouds.recenter_cloud(xyz)
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(*np.rollaxis(xyz, 1), c=color, s=s, cmap=cmap)
    ax.set_aspect("equal")
    return ax
