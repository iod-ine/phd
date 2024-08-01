import matplotlib.pyplot as plt
import numpy as np


def scatter_point_cloud_3d(
    xyz: np.ndarray,
    *,
    color=None,
    figsize=None,
    elev=None,
    azim=None,
) -> plt.Axes:
    color = color if color is not None else xyz[:, 2]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(*np.rollaxis(xyz, 1), c=color, s=1)
    ax.set_aspect("equal")
    return ax
