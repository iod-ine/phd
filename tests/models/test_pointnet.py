"""Tests for the PointNet++ model code."""

import pytest
import torch
import torch_geometric

from src.models.pointnet.semantic_segmentation import PointNet2SemanticSegmentor
from tests.models.utils import assert_parameters_change_after_learning_step

# TODO: Add tests for parameter changing after step, loss decreasing after step.


@pytest.fixture(scope="function")
def model():
    """An instance of the PointNet++ segmentor."""
    return PointNet2SemanticSegmentor(num_features=5, num_classes=10)


@pytest.fixture(scope="function")
def example():
    """An example point cloud for testing."""
    return torch_geometric.data.Data(
        x=torch.rand(size=(10, 5)),
        pos=torch.rand(size=(10, 3)),
        y=torch.randint(low=0, high=10, size=(10,)),
        batch=torch.zeros(size=(10,), dtype=torch.int64),
    )


def test_parameters_change_after_learning_step(model, example):
    """If the parameters are not frozen, they should change after a learning step."""
    assert_parameters_change_after_learning_step(
        model,
        torch.nn.CrossEntropyLoss(),
        example,
    )
