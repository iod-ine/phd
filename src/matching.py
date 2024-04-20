import numpy as np
import scipy


def match_candidates(
    ground_truth: np.ndarray,
    candidates: np.ndarray,
    *,
    max_distance: float,
    max_height_difference: float,
) -> list[dict]:
    """Match ground truth trees to candidates.

    Args:
        ground_truth (np.ndarray): Array of shape (N, 2) with positions of the ground
            truth trees.
        candidates (np.ndarray): Array of shape (M, 3) with positions and heights of
            detected trees.
        max_distance (float): Maximum distance between actual and potential trees to
            consider them a matched pair.
        max_height_difference (float): Maximum height difference between actual and
            potential trees to consider them a matched pair.

    """

    distance_matrix = scipy.spatial.distance_matrix(
        x=ground_truth[:, :2],
        y=candidates[:, :2],
    )
    indices = np.nonzero(distance_matrix <= max_distance)  # (ground_truths, candidates)
    distances = distance_matrix[indices]
    sparse_distances = sorted((pair, d) for pair, d in zip(zip(*indices), distances))

    ground_truth_matched_mask = np.zeros(ground_truth.shape[0], dtype=bool)
    candidates_matched_mask = np.zeros(candidates.shape[0], dtype=bool)
    out = []

    def ndarray_to_tuple(array: np.ndarray) -> tuple:
        return tuple(None if np.isnan(x) else x for x in array)

    for (i, j), distance in sparse_distances:
        if (
            np.isnan(ground_truth[i][2])
            or abs(ground_truth[i][2] - candidates[j][2]) <= max_height_difference
        ):
            if ground_truth_matched_mask[i]:
                candidates_matched_mask[j] = True
                out.append(
                    {
                        "ground_truth": None,
                        "candidate": ndarray_to_tuple(candidates[j]),
                        "class": "FP",
                        "distance": None,
                    }
                )
            elif candidates_matched_mask[j]:
                ground_truth_matched_mask[i] = True
                out.append(
                    {
                        "ground_truth": ndarray_to_tuple(ground_truth[i]),
                        "candidate": None,
                        "class": "FN",
                        "distance": None,
                    }
                )
            else:
                ground_truth_matched_mask[i] = True
                candidates_matched_mask[j] = True
                out.append(
                    {
                        "ground_truth": ndarray_to_tuple(ground_truth[i]),
                        "candidate": ndarray_to_tuple(candidates[j]),
                        "class": "TP",
                        "distance": distance,
                    }
                )

    out.extend(
        {
            "ground_truth": tuple(point),
            "candidate": None,
            "class": "FN",
            "distance": None,
        }
        for point in ground_truth[~ground_truth_matched_mask]
    )

    out.extend(
        {
            "ground_truth": None,
            "candidate": tuple(point),
            "class": "FP",
            "distance": None,
        }
        for point in candidates[~candidates_matched_mask]
    )

    return out
