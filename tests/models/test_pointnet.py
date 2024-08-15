import pytest
import torch
import torch_geometric

from src.models.pointnet.semantic_segmentation import PointNet2Segmentor
from tests.models.utils import assert_parameters_change_after_learning_step

# TODO: Add tests for parameter changing after step, loss decreasing after step.


@pytest.fixture(scope="function")
def model():
    """An instance of the PointNet++ segmentor."""
    return PointNet2Segmentor(n_features=4)


@pytest.fixture(scope="function")
def example():
    """An example point cloud for testing."""
    return torch_geometric.data.Data(
        x=torch.rand((100, 4), dtype=torch.float32),
        pos=torch.rand((100, 3), dtype=torch.float32),
        y=torch.rand(100, dtype=torch.float32),
        batch=torch.zeros(100, dtype=torch.int64),
    )


def test_parameters_change_after_learning_step(model, example):
    """If the parameters are not frozen, they should change after a learning step."""
    assert_parameters_change_after_learning_step(model, example)
