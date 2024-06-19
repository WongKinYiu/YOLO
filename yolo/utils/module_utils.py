import inspect
from typing import Tuple, Union

from torch import Tensor, nn
from torch.nn.common_types import _size_2_t


def get_layer_map():
    """
    Dynamically generates a dictionary mapping class names to classes,
    filtering to include only those that are subclasses of nn.Module,
    ensuring they are relevant neural network layers.
    """
    layer_map = {}
    from yolo.model import module

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            layer_map[name] = obj
    return layer_map


def auto_pad(kernel_size: _size_2_t, dilation: _size_2_t = 1, **kwargs) -> Tuple[int, int]:
    """
    Auto Padding for the convolution blocks
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def create_activation_function(activation: str) -> nn.Module:
    """
    Retrieves an activation function from the PyTorch nn module based on its name, case-insensitively.
    """
    if not activation or activation.lower() in ["false", "none"]:
        return nn.Identity()

    activation_map = {
        name.lower(): obj
        for name, obj in nn.modules.activation.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, nn.Module)
    }
    if activation.lower() in activation_map:
        return activation_map[activation.lower()](inplace=True)
    else:
        raise ValueError(f"Activation function '{activation}' is not found in torch.nn")


def round_up(x: Union[int, Tensor], div: int = 1) -> Union[int, Tensor]:
    """
    Rounds up `x` to the bigger-nearest multiple of `div`.
    """
    return x + (-x % div)


def divide_into_chunks(input_list, chunk_num):
    """
    Args: input_list: [0, 1, 2, 3, 4, 5], chunk: 2
    Return: [[0, 1, 2], [3, 4, 5]]
    """
    list_size = len(input_list)

    if list_size % chunk_num != 0:
        raise ValueError(
            f"The length of the input list ({list_size}) must be exactly divisible by the number of chunks ({chunk_num})."
        )

    chunk_size = list_size // chunk_num
    return [input_list[i : i + chunk_size] for i in range(0, list_size, chunk_size)]
