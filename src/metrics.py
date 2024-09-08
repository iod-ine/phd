"""Implementations of metrics."""

import torch
import torchmetrics


class Accuracy(torchmetrics.Metric):
    """Proportion of points, for which the prediction is correct."""

    def __init__(self):
        """Create a new metric instance."""
        super().__init__()
        self.add_state(name="correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(name="total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update the internal state."""
        self.correct += torch.sum(pred == target)
        self.total += target.numel()

    def compute(self):
        """Use the internal state to compute the final value."""
        return self.correct / self.total


class IntersectionOverUnion(torchmetrics.Metric):
    """Micro-averaged intersection over union for tree segmentation."""

    def __init__(self):
        """Create a new metric instance."""
        super().__init__()
        self.add_state(name="intersection", default=torch.tensor(0))
        self.add_state(name="union", default=torch.tensor(0))

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update the internal state."""
        self.intersection += torch.sum(pred == target)
        for i in target.unique():
            self.union += torch.sum((pred == i) | (target == i))

    def compute(self):
        """Use the internal state to compute the final value."""
        return self.intersection / self.union
