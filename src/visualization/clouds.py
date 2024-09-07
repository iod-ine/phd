"""Functions for visualizing point clouds."""

import matplotlib.pyplot as plt
import numpy as np


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
