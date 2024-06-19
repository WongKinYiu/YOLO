import re
import sys
from pathlib import Path

import pytest
from torch import nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from yolo.utils.module_utils import (
    auto_pad,
    create_activation_function,
    divide_into_chunks,
)


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
    result = create_activation_function(activation_name)
    assert isinstance(result, expected_type), f"get_activation does not return correct type for {activation_name}"


def test_get_activation_invalid():
    with pytest.raises(ValueError):
        create_activation_function("unsupported_activation")


def test_divide_into_chunks():
    input_list = [0, 1, 2, 3, 4, 5]
    chunk_num = 2
    expected_output = [[0, 1, 2], [3, 4, 5]]
    assert divide_into_chunks(input_list, chunk_num) == expected_output


def test_divide_into_chunks_non_divisible_length():
    input_list = [0, 1, 2, 3, 4, 5]
    chunk_num = 4
    with pytest.raises(
        ValueError,
        match=re.escape("The length of the input list (6) must be exactly divisible by the number of chunks (4)."),
    ):
        divide_into_chunks(input_list, chunk_num)


def test_divide_into_chunks_single_chunk():
    input_list = [0, 1, 2, 3, 4, 5]
    chunk_num = 1
    expected_output = [[0, 1, 2, 3, 4, 5]]
    assert divide_into_chunks(input_list, chunk_num) == expected_output


def test_divide_into_chunks_equal_chunks():
    input_list = [0, 1, 2, 3, 4, 5, 6, 7]
    chunk_num = 4
    expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert divide_into_chunks(input_list, chunk_num) == expected_output
