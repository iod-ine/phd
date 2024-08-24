"""Test functions for manipulating point clouds."""

import numpy as np
import pytest

import src.clouds


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0, id="no_height_threshold"),
        pytest.param(0.1, id="with_height_threshold"),
    ],
)
def test_create_regular_grid_shapes_match(threshold):
    """Make sure that the shapes returned by create_grid are compatible."""
    rng = np.random.default_rng(seed=42)

    xyzs = [rng.random(size=(10 * i, 3)) for i in range(1, 4)]
    las_list = [src.clouds.numpy_to_las(xyz) for xyz in xyzs]

    pos, x, y = src.clouds.create_regular_grid(
        las_list=las_list,
        ncols=3,
        dx=1,
        dy=1,
        height_threshold=threshold,
    )

    assert pos.shape[0] == x.shape[0] == y.shape[0]
