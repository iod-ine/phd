import numpy as np
import pytest
import shapely

import src.utils


@pytest.mark.parametrize(
    "other,expected",
    [
        pytest.param(
            np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
            np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
            id="middle",
        ),
        pytest.param(
            np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            id="bottom-left",
        ),
        pytest.param(
            np.array([[0, 0], [2, 2], [3, 2], [1, 0]]),
            np.array([[0, 0], [1, 1], [2, 2], [1, 0], [2, 1], [3, 2]]),
            id="skewed",
        ),
    ],
)
def test_crop_by_other(other, expected):
    points = np.empty((16, 2), dtype=int)
    points[:, 0] = np.repeat([0, 1, 2, 3], 4)
    points[:, 1] = np.tile([0, 1, 2, 3], 4)

    result = src.utils.crop_by_other(points, other)

    assert np.all(np.sort(result, axis=0) == np.sort(expected, axis=0))


def test_extract_points_from_matches(matches):
    generator = src.utils.extract_points_from_matches(matches["first"])
    assert next(generator) == {
        "geometry": shapely.Point(0, 0),
        "class": "TP_gt",
        "height": 5,
    }
    assert next(generator) == {
        "geometry": shapely.Point(0, 1),
        "class": "TP",
        "height": 3,
    }
    assert next(generator) == {
        "geometry": shapely.Point(0, 2),
        "class": "FP",
        "height": 4,
    }
    with pytest.raises(StopIteration):
        next(generator)
