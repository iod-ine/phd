import numpy as np
import pytest

from src.lmf import local_maxima_filter


@pytest.fixture()
def point_cloud():
    """A tiny handmade point cloud with two local maxima of different heights.

    Here is how it looks like (the numbers represent heights and the distances between neighboring points are 1):
        2  0  0
        0  1  0
        0  0  3

    """
    cloud = np.zeros((9, 3))
    cloud[:, 0] = np.repeat(np.arange(3), 3)
    cloud[:, 1] = np.tile(np.arange(3), 3)
    cloud[0, 2] = 2
    cloud[4, 2] = 1
    cloud[8, 2] = 3
    return cloud


def test_lmf_finds_correct_maxima(point_cloud):
    """The function should correctly identify local maxima based in the parameters."""
    detected_maxima = local_maxima_filter(point_cloud, window_size=1, height_threshold=0)
    assert np.all(detected_maxima == np.array([[0, 0, 2], [1, 1, 1], [2, 2, 3]]))

    detected_maxima = local_maxima_filter(point_cloud, window_size=2, height_threshold=0)
    assert np.all(detected_maxima == np.array([[0, 0, 2], [2, 2, 3]]))

    detected_maxima = local_maxima_filter(point_cloud, window_size=10, height_threshold=0)
    assert np.all(detected_maxima == np.array([[2, 2, 3]]))

    detected_maxima = local_maxima_filter(point_cloud, window_size=1, height_threshold=1)
    assert np.all(detected_maxima == np.array([[0, 0, 2], [2, 2, 3]]))

    detected_maxima = local_maxima_filter(point_cloud, window_size=1, height_threshold=2)
    assert np.all(detected_maxima == np.array([[2, 2, 3]]))

    detected_maxima = local_maxima_filter(point_cloud, window_size=1, height_threshold=5)
    assert detected_maxima.size == 0


def test_lmf_rejects_non_numpy_input():
    """The function assumes its input is a Numpy array and should reject anything else."""

    with pytest.raises(TypeError):
        local_maxima_filter([[0, 0, 0], [1, 1, 1]], window_size=1, height_threshold=0)
