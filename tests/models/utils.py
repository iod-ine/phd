"""Helper functions to test models."""

from typing import Callable

import torch
import torch_geometric


def assert_parameters_change_after_learning_step(
    model: torch.nn.Module,
    criterion: Callable,
    example: torch_geometric.data.Data,
):
    """If the parameters are not frozen, they should change after a learning step."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    params_before = [
        (name, data.detach().clone()) for name, data in model.named_parameters()
    ]

    pred = model(example)
    loss = criterion(pred, example.y)
    loss.backward()
    optimizer.step()

    params_after = model.parameters()
    for before, after in zip(params_before, params_after):
        assert torch.any(before[1] != after), f"Parameter {before[0]} did not change"
