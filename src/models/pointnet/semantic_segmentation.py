"""The implementation of a PointNet++ for semantic segmentation."""

import torch
import torch_geometric

from src.models.pointnet.utils import SetAbstraction


class PointNet2SemanticSegmentor(torch.nn.Module):
    """PointNet++ model for semantic segmentation."""

    def __init__(self, num_features: int, num_classes: int):
        """Create a new PointNet++ semantic segmentor."""
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.set_abstraction_0 = SetAbstraction(
            ratio=0.5,
            r=0.2,
            local_nn=torch_geometric.nn.MLP(
                channel_list=[num_features + 3, 64, 128, 256],
            ),
        )
        self.set_abstraction_1 = SetAbstraction(
            ratio=0.25,
            r=0.4,
            local_nn=torch_geometric.nn.MLP(
                channel_list=[256 + 3, 512, 1024, 2048],
            ),
        )
        self.unit_point_net_0 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[2048 + 256 + 3, 2048, 1024, 512],
                bias=False,
            ),
            add_self_loops=True,
        )
        self.unit_point_net_1 = torch_geometric.nn.PointNetConv(
            local_nn=torch_geometric.nn.MLP(
                channel_list=[512 + num_features + 3, 512, 256, 128],
                bias=False,
            ),
            add_self_loops=True,
        )
        self.classifier = torch_geometric.nn.MLP(
            channel_list=[128, 64, 32, num_classes],
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
            edge_index=torch.empty((2, 0), dtype=torch.int64),
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
            edge_index=torch.empty((2, 0), dtype=torch.int64),
        )
        return self.classifier(x_3)


if __name__ == "__main__":
    s3dis = torch_geometric.datasets.S3DIS(root="data/external/S3DIS/")
    loader = torch_geometric.loader.DataLoader(s3dis, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    model = PointNet2SemanticSegmentor(num_features=6, num_classes=13)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("Overfitting a single batch:")

    model.train()
    for i in range(20):
        pred = model(batch)
        loss = criterion(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  At iteration {i:>02} loss is {loss.item():.4f}")
