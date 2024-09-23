"""Implementation of the per tree reverse distance weighted loss."""

from typing import Literal

import torch
import torch_scatter
from torch import Tensor


class PerTreeReverseDistanceWeighted(torch.nn.Module):
    """Modifies the base loss by reverse distance weighting per tree."""

    def __init__(self, base_loss: Literal["l1", "l2"]) -> None:
        """Create a new instance."""
        super().__init__()
        if base_loss == "l1":
            self.base_loss = torch.nn.L1Loss(reduction="none")
        elif base_loss == "l2":
            self.base_loss = torch.nn.MSELoss(reduction="none")

    def forward(self, prediction: Tensor, target: Tensor, pos: Tensor) -> Tensor:
        """Compute the base loss and adjust it by the distance to the tree centroid."""
        base_loss = self.base_loss(prediction, target)
        target = target.to(torch.int64)
        tree_centroids = torch_scatter.scatter_mean(
            src=pos,
            index=target,
            dim=0,
        )
        distance_to_tree_centroid = torch.norm(
            pos[:, :2] - tree_centroids[target, :2],
            p=2,
            dim=1,
        )
        return base_loss / (distance_to_tree_centroid + 1)
