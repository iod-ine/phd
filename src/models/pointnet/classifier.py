"""The implementation of a PointNet++ classifier."""

import torch
import torch_geometric

from src.models.pointnet.utils import GlobalSetAbstraction, SetAbstraction


class PointNet2Classifier(torch.nn.Module):
    """PointNet++ classifier."""

    def __init__(self, num_classes: int, num_features: int):
        """Create a new PointNet++ classifier."""
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            ratio=0.5,
            r=0.2,
            local_nn=torch_geometric.nn.MLP([3 + num_features, 64, 64, 128]),
        )
        self.sa2_module = SetAbstraction(
            0.25,
            0.4,
            torch_geometric.nn.MLP([128 + 3, 128, 128, 256]),
        )
        self.sa3_module = GlobalSetAbstraction(
            nn=torch_geometric.nn.MLP([256 + 3, 256, 512, 1024]),
        )

        self.mlp = torch_geometric.nn.MLP(
            channel_list=[1024, 512, 256, num_classes],
            dropout=0.5,
            norm=None,
        )

    def forward(self, data):
        """A forward pass through the network."""
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)
