import inspect

import torch.nn as nn

from model import module


def auto_pad(kernel_size, dilation):
    """
    Calculate padding size for convolutional operations given kernel size and dilation.

    :param kernel_size: An int or tuple of two ints indicating the kernel dimensions.
    :param dilation: An int or tuple of two ints indicating the dilation in height and width.
    :return: An int or tuple of ints representing the padding required.
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    effective_kernel_h = dilation[0] * (kernel_size[0] - 1) + 1
    effective_kernel_w = dilation[1] * (kernel_size[1] - 1) + 1
    padding_h = effective_kernel_h // 2
    padding_w = effective_kernel_w // 2

    if padding_h == padding_w:
        return padding_h
    return (padding_h, padding_w)


def get_layer_map():
    """
    Dynamically generates a dictionary mapping class names to classes,
    filtering to include only those that are subclasses of nn.Module,
    ensuring they are relevant neural network layers.
    """
    layer_map = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            layer_map[name] = obj
    return layer_map
