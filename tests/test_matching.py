import numpy as np
import pytest

from src.matching import match_candidates

matches = {
    "first": [
        {
            "ground_truth": (0, 0, 5),
            "candidate": (0, 1, 3),
            "class": "TP",
            "distance": 1.0,
        },
        {
            "ground_truth": None,
            "candidate": (0, 2, 4),
            "class": "FP",
            "distance": None,
        },
    ],
    "first_nanh": [
        {
            "ground_truth": (0, 0, None),
            "candidate": (0, 1, 3),
            "class": "TP",
            "distance": 1.0,
        },
        {
            "ground_truth": None,
            "candidate": (0, 2, 4),
            "class": "FP",
            "distance": None,
        },
    ],
    "second": [
        {
            "ground_truth": (0, 0, 5),
            "candidate": (0, 2, 4),
            "class": "TP",
            "distance": 2.0,
        },
        {
            "ground_truth": None,
            "candidate": (0, 1, 3),
            "class": "FP",
            "distance": None,
        },
    ],
    "none": [
        {
            "ground_truth": (0, 0, 5),
            "candidate": None,
            "class": "FN",
            "distance": None,
        },
        {
            "ground_truth": None,
            "candidate": (0, 1, 3),
            "class": "FP",
            "distance": None,
        },
        {
            "ground_truth": None,
            "candidate": (0, 2, 4),
            "class": "FP",
            "distance": None,
        },
    ],
}


def _matches_are_equal(first, second):
    """A utility function to compare lists of candidates regardless of their order."""

    def get_coord_tuple(match):
        candidate = match["candidate"] or (0, 0, 0)
        ground_truth = match["ground_truth"] or (0, 0, 0)
        if ground_truth[-1] is None:
            ground_truth = (ground_truth[0], ground_truth[1], -1)
        return (*ground_truth, *candidate)

    return sorted(first, key=get_coord_tuple) == sorted(second, key=get_coord_tuple)


def test_utility_function_matches_are_equal():
    for match in matches.values():
        assert _matches_are_equal(match, match[::-1])


@pytest.mark.parametrize(
    "max_distance,max_height_difference,expected",
    [
        pytest.param(
            5,
            5,
            matches["first"],
            id="all_within",
        ),
        pytest.param(
            5,
            1,
            matches["second"],
            id="height_threshold_one",
        ),
        pytest.param(
            5,
            0,
            matches["none"],
            id="height_threshold_all",
        ),
        pytest.param(
            1,
            5,
            matches["first"],
            id="distance_threshold_one",
        ),
        pytest.param(
            0,
            5,
            matches["none"],
            id="distance_threshold_all",
        ),
    ],
)
def test_match_candidates_thresholds(
    max_distance,
    max_height_difference,
    expected,
):
    actual = match_candidates(
        ground_truth=np.array([[0, 0, 5]]),
        candidates=np.array([[0, 1, 3], [0, 2, 4]]),
        max_distance=max_distance,
        max_height_difference=max_height_difference,
    )
    assert _matches_are_equal(actual, expected)


def test_match_candidates_with_nan_height():
    actual = match_candidates(
        ground_truth=np.array([[0, 0, np.nan]]),
        candidates=np.array([[0, 1, 3], [0, 2, 4]]),
        max_distance=5,
        max_height_difference=5,
    )
    assert _matches_are_equal(actual, matches["first_nanh"])
