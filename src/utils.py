"""Utility functions."""

from typing import Generator

import numpy as np
import scipy.spatial
import shapely


def crop_by_other(points: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Crop points by the extent of other."""
    hull = scipy.spatial.ConvexHull(other[:, :2])
    vertex_points = hull.points[hull.vertices]
    delaunay = scipy.spatial.Delaunay(vertex_points)
    within_hull = delaunay.find_simplex(points[:, :2]) >= 0
    return points[within_hull]


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
