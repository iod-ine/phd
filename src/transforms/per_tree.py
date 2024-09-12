"""Transforms applying to individual trees within a synthetic forest patch."""

import math
import random

import torch
import torch_geometric


class PerTreeRandomRotateScale(torch_geometric.transforms.BaseTransform):
    """Randomly rotate and scale each tree within a synthetic forest patch."""

    def __init__(self, scales: tuple[float, float], degrees: tuple[float, float]):
        """Initialize the transform."""
        self.scales = scales
        self.degrees = degrees

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Apply the transform."""
        for i in data.y.unique():
            mask = data.y == i
            tree = data.pos[mask]
            means = tree.mean(dim=0, keepdims=True)
            scale = random.uniform(*self.scales)
            angle = math.radians(random.uniform(*self.degrees))
            sin, cos = math.sin(angle), math.cos(angle)
            rotation_matrix = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
            data.pos[mask] = ((tree - means) @ rotation_matrix + means) * scale
        return data
