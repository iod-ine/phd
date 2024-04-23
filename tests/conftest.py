import pytest


@pytest.fixture()
def matches():
    return {
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
