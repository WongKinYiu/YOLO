import inspect

import torch.nn as nn

from model import module


def auto_pad():
    raise NotImplementedError


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
