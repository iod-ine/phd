"""The implementation of a PointNet++ for semantic segmentation."""

import torch
import torch_geometric

from src.models.pointnet.utils import SetAbstraction


class PointNet2Segmentor(torch.nn.Module):
    """PointNet++ segmentor."""

    def __init__(self, n_features: int):
        """Create a new PointNet++ segmentor."""
        super().__init__()
        self.n_features = n_features
        self.set_abstraction_0 = SetAbstraction(
            ratio=0.5,
            r=0.2,
            local_nn=torch_geometric.nn.MLP([self.n_features + 3, 64, 128, 256]),
        )
        self.set_abstraction_1 = SetAbstraction(
            ratio=0.25,
            r=0.4,
            local_nn=torch_geometric.nn.MLP([256 + 3, 512, 1024, 1024]),
        )
        self.unit_point_net_0 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[1024 + 256 + 3, 1024, 512, 256],
                bias=False,
            ),
            add_self_loops=False,
        )
        self.unit_point_net_1 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[256 + self.n_features + 3, 256, 128, 64, 1],
                bias=False,
            ),
            add_self_loops=False,
        )

    # TODO: Figure out better names for variables within forward()
    # TODO: Switch to add_self_loops=True in unit PointNets maybe?..
    def forward(self, data):
        """A forward pass through the network."""
        x, pos, batch = data.x, data.pos, data.batch
        x_0, pos_0, batch_0 = self.set_abstraction_0(x, pos, batch)
        x_1, pos_1, batch_1 = self.set_abstraction_1(x_0, pos_0, batch_0)
        x_1_interpolated = torch_geometric.nn.knn_interpolate(
            x=x_1,
            pos_x=pos_1,
            pos_y=pos_0,
            batch_x=batch_1,
            batch_y=batch_0,
            k=3,
        )
        x_2 = torch.cat([x_1_interpolated, x_0], dim=1)
        edge_index = torch_geometric.nn.knn_graph(  # self-loops only
            x=x_2,
            k=1,
            batch=batch_0,
            loop=True,
        )
        x_3 = self.unit_point_net_0(
            x=x_2,
            pos=pos_0,
            edge_index=edge_index,
        )
        x_3_interpolated = torch_geometric.nn.knn_interpolate(
            x=x_3,
            pos_x=pos_0,
            pos_y=pos,
            batch_x=batch_0,
            batch_y=batch,
            k=3,
        )
        x_4 = torch.cat([x_3_interpolated, x], dim=1)
        edge_index = torch_geometric.nn.knn_graph(  # self-loops only
            x=x_4,
            k=1,
            batch=batch,
            loop=True,
        )
        out = self.unit_point_net_1(
            x=x_4,
            pos=pos,
            edge_index=edge_index,
        )

        # TODO: What to return here?
        return torch_geometric.data.Data(
            pred=out,
            pos=pos,
            batch=batch,
        )
