"""The implementation of a PointNet++ for semantic segmentation."""

import torch
import torch_geometric

from src.models.pointnet.utils import SetAbstraction


class PointNet2TreeSegmentor(torch.nn.Module):
    """PointNet++ model for tree segmentation.

    It's like instance segmentation but the only class is a tree.

    """

    def __init__(
        self,
        num_features: int,
        set_abstraction_ratios: tuple[float, float] = (0.5, 0.25),
        set_abstraction_radii: tuple[float, float] = (0.2, 0.4),
    ):
        """Create a new PointNet++ semantic segmentor."""
        super().__init__()

        self.num_features = num_features
        self.set_abstraction_0 = SetAbstraction(
            ratio=set_abstraction_ratios[0],
            r=set_abstraction_radii[0],
            local_nn=torch_geometric.nn.MLP(
                channel_list=[num_features + 3, 64, 128, 256],
            ),
        )
        self.set_abstraction_1 = SetAbstraction(
            ratio=set_abstraction_ratios[1],
            r=set_abstraction_radii[1],
            local_nn=torch_geometric.nn.MLP(
                channel_list=[256 + 3, 512, 1024, 2048],
            ),
        )
        self.unit_point_net_0 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[2048 + 256 + 3, 1024, 512, 256],
                bias=False,
            ),
            add_self_loops=True,
        )
        self.unit_point_net_1 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[256 + num_features + 3, 256, 128, 64],
                bias=False,
            ),
            add_self_loops=True,
        )
        self.regressor = torch_geometric.nn.MLP(
            channel_list=[64 + 3, 32, 16, 1],
            bias=False,
        )

    def forward(self, data):
        """A forward pass through the network."""
        x_in, pos_in, batch_in = data.x, data.pos, data.batch
        x_0, pos_0, batch_0 = self.set_abstraction_0(x_in, pos_in, batch_in)
        x_1, pos_1, batch_1 = self.set_abstraction_1(x_0, pos_0, batch_0)
        x_1_interpolated = torch_geometric.nn.knn_interpolate(
            x=x_1,
            pos_x=pos_1,
            pos_y=pos_0,
            batch_x=batch_1,
            batch_y=batch_0,
            k=3,
        )
        x_2 = self.unit_point_net_0(
            x=torch.cat([x_1_interpolated, x_0], dim=1),
            pos=pos_0,
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=x_in.device),
        )
        x_2_interpolated = torch_geometric.nn.knn_interpolate(
            x=x_2,
            pos_x=pos_0,
            pos_y=pos_in,
            batch_x=batch_0,
            batch_y=batch_in,
            k=3,
        )
        x_3 = self.unit_point_net_1(
            x=torch.cat([x_2_interpolated, x_in], dim=1),
            pos=pos_in,
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=x_in.device),
        )
        return self.regressor(torch.cat([x_3, pos_in], dim=1))


class PointNet2TreeSegmentorLarge(torch.nn.Module):
    """Large PointNet++ model for tree segmentation.

    It's like instance segmentation but the only class is a tree.

    """

    def __init__(
        self,
        num_features: int,
        set_abstraction_ratios: tuple[float, float] = (0.5, 0.5, 0.5),
        set_abstraction_radii: tuple[float, float] = (0.4, 0.4, 0.4),
    ):
        """Create a new PointNet++ semantic segmentor."""
        super().__init__()

        self.num_features = num_features
        self.set_abstraction_0 = SetAbstraction(
            ratio=set_abstraction_ratios[0],
            r=set_abstraction_radii[0],
            local_nn=torch_geometric.nn.MLP(
                channel_list=[num_features + 3, 64, 128, 256],
            ),
        )
        self.set_abstraction_1 = SetAbstraction(
            ratio=set_abstraction_ratios[1],
            r=set_abstraction_radii[1],
            local_nn=torch_geometric.nn.MLP(
                channel_list=[256 + 3, 512, 1024, 2048],
            ),
        )
        self.set_abstraction_2 = SetAbstraction(
            ratio=set_abstraction_ratios[2],
            r=set_abstraction_radii[2],
            local_nn=torch_geometric.nn.MLP(
                channel_list=[2048 + 3, 2048, 2048, 2048],
            ),
        )
        self.unit_point_net_0 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[2048 + 2048 + 3, 2048, 1024, 512],
                bias=False,
            ),
            add_self_loops=True,
        )
        self.unit_point_net_1 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[512 + 256 + 3, 2048, 1024, 512],
                bias=False,
            ),
            add_self_loops=True,
        )
        self.unit_point_net_2 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[512 + num_features + 3, 256, 128, 64],
                bias=False,
            ),
            add_self_loops=True,
        )
        self.regressor = torch_geometric.nn.MLP(
            channel_list=[64 + 3, 32, 16, 1],
            bias=False,
        )

    def forward(self, data):
        """A forward pass through the network."""
        x_in, pos_in, batch_in = data.x, data.pos, data.batch
        x_0, pos_0, batch_0 = self.set_abstraction_0(x_in, pos_in, batch_in)
        x_1, pos_1, batch_1 = self.set_abstraction_1(x_0, pos_0, batch_0)
        x_2, pos_2, batch_2 = self.set_abstraction_2(x_1, pos_1, batch_1)
        interpolated = torch_geometric.nn.knn_interpolate(
            x=x_2,
            pos_x=pos_2,
            pos_y=pos_1,
            batch_x=batch_2,
            batch_y=batch_1,
            k=3,
        )
        x_2 = self.unit_point_net_0(
            x=torch.cat([interpolated, x_1], dim=1),
            pos=pos_1,
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=x_in.device),
        )
        interpolated = torch_geometric.nn.knn_interpolate(
            x=x_2,
            pos_x=pos_1,
            pos_y=pos_0,
            batch_x=batch_1,
            batch_y=batch_0,
            k=3,
        )
        x_2 = self.unit_point_net_1(
            x=torch.cat([interpolated, x_0], dim=1),
            pos=pos_0,
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=x_in.device),
        )
        interpolated = torch_geometric.nn.knn_interpolate(
            x=x_2,
            pos_x=pos_0,
            pos_y=pos_in,
            batch_x=batch_0,
            batch_y=batch_in,
            k=3,
        )
        x_2 = self.unit_point_net_2(
            x=torch.cat([interpolated, x_in], dim=1),
            pos=pos_in,
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=x_in.device),
        )
        return self.regressor(torch.cat([x_2, pos_in], dim=1))


if __name__ == "__main__":
    import src.datasets
    dataset = src.datasets.SyntheticForestRGBMBFPatch(
        "data/interim/synthetic_forest/",
        patch_width=5,
        patch_height=5,
        force_reload=False,
    )
    example = dataset[0]
    example["batch"] = torch.zeros_like(example.y)
    model = PointNet2TreeSegmentorLarge(33)
    pred = model(example)
