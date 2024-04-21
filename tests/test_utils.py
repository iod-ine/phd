import numpy as np
import pytest

import src.utils


@pytest.mark.parametrize(
    "other,expected",
    [
        pytest.param(
            np.array([[1, 1], [2, 2]]),
            np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
            id="middle",
        ),
        pytest.param(
            np.array([[-1, -1], [1, 1]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            id="bottom-left",
        ),
    ],
)
def test_crop_by_other(other, expected):
    points = np.empty((16, 2), dtype=int)
    points[:, 0] = np.repeat([0, 1, 2, 3], 4)
    points[:, 1] = np.tile([0, 1, 2, 3], 4)

    result = src.utils.crop_by_other(points, other)

    assert np.all(result == expected)
