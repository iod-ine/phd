"""Utility functions."""

from typing import Generator

import numpy as np
import shapely


def crop_by_other(points: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Crop points by the extent of other."""
    crop_indices = np.nonzero(
        (points[:, 0] >= other[:, 0].min())
        & (points[:, 1] >= other[:, 1].min())
        & (points[:, 0] <= other[:, 0].max())
        & (points[:, 1] <= other[:, 1].max())
    )
    return points[crop_indices]


def extract_points_from_matches(matches: dict) -> Generator[dict, None, None]:
    """Generate points from matches to load into geopandas."""
    for match in matches:
        if match["ground_truth"] is not None:
            yield {
                "geometry": shapely.Point(match["ground_truth"][:2]),
                "class": "TP_gt" if match["class"] == "TP" else match["class"],
                "height": match["ground_truth"][2],
            }
        if match["candidate"] is not None:
            yield {
                "geometry": shapely.Point(match["candidate"][:2]),
                "class": "TP" if match["class"] == "TP" else match["class"],
                "height": match["candidate"][2],
            }
