import sys
from pathlib import Path

import pytest
import torch
from torch import nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from yolo.tools.module_helper import auto_pad, get_activation


@pytest.mark.parametrize(
    "kernel_size, dilation, expected",
    [
        (3, 1, (1, 1)),
        ((3, 3), (1, 1), (1, 1)),
        (3, (2, 2), (2, 2)),
        ((5, 5), 1, (2, 2)),
        ((3, 5), (2, 1), (2, 2)),
    ],
)
def test_auto_pad(kernel_size, dilation, expected):
    assert auto_pad(kernel_size, dilation) == expected, "auto_pad does not calculate padding correctly"


@pytest.mark.parametrize(
    "activation_name, expected_type",
    [("ReLU", nn.ReLU), ("leakyrelu", nn.LeakyReLU), ("none", nn.Identity), (None, nn.Identity), (False, nn.Identity)],
)
def test_get_activation(activation_name, expected_type):
    result = get_activation(activation_name)
    assert isinstance(result, expected_type), f"get_activation does not return correct type for {activation_name}"


def test_get_activation_invalid():
    with pytest.raises(ValueError):
        get_activation("unsupported_activation")
