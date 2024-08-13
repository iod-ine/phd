"""Implementations of helper modules used in the PointNet++."""

from typing import Callable

import torch
import torch_geometric


class SetAbstraction(torch.nn.Module):
    """Set abstraction module from the PointNet++ paper.

    Implementation taken from torch_geometric documentation and examples in the
    repository, with slight reformatting.
    """

    def __init__(
        self,
        ratio: float,
        r: float,
        local_nn: Callable,
        max_num_neighbors: int = 64,
    ):
        """Create a new SetAbstraction instance.

        Args:
            ratio: Ratio of points to sample (using farthest point sampling (FPS)).
            r: Radius used to collect neighborhoods of sampled points.
            local_nn: A neural network that processes node features and coordinates.
            max_num_neighbors: Maximum number of neighbors collected after sampling.
        """
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.conv = torch_geometric.nn.PointNetConv(
            local_nn=local_nn,
            add_self_loops=False,
        )

    def forward(self, x, pos, batch):
        """A forward pass through the module."""
        idx = torch_geometric.nn.fps(pos, batch, ratio=self.ratio)
        row, col = torch_geometric.nn.radius(
            x=pos,
            y=pos[idx],
            r=self.r,
            batch_x=batch,
            batch_y=batch[idx],
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSetAbstraction(torch.nn.Module):
    """Global set abstraction module from the PointNet++ paper.

    Reduces the input point cloud into a single feature vector.

    Implementation taken from torch_geometric documentation and examples in the
    repository, with slight reformatting.
    """

    def __init__(self, nn: Callable):
        """Create a new GlobalSetAbstraction instance.

        Args:
            nn: A neural network that processes node features and coordinates.
        """
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        """A forward pass through the module."""
        x = self.nn(torch.cat([x, pos], dim=1))
        x = torch_geometric.nn.global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
