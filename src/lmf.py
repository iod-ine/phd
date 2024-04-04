"""Local maxima filtering."""

import numpy as np
import scipy


def local_maxima_filter(
    cloud: np.ndarray,
    window_size: float,
    height_threshold: float,
) -> np.ndarray:
    """Detect local maxima in the point cloud with a fixed window size."""

    if not isinstance(cloud, np.ndarray):
        raise TypeError(f"Cloud needs to be a numpy array, not {type(cloud)}")

    cloud = cloud[cloud[:, 2] > height_threshold]
    tree = scipy.spatial.KDTree(data=cloud)
    seen_mask = np.zeros(cloud.shape[0], dtype=bool)
    local_maxima = []

    for i, point in enumerate(cloud):
        if seen_mask[i]:
            continue
        neighbor_indices = tree.query_ball_point(point, window_size)
        highest_neighbor = neighbor_indices[cloud[neighbor_indices, 2].argmax()]
        seen_mask[neighbor_indices] = True
        seen_mask[highest_neighbor] = False

        if i == highest_neighbor:
            local_maxima.append(i)

    return cloud[local_maxima]
