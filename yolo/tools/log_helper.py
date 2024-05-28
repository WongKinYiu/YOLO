"""
Module for initializing logging tools used in machine learning and data processing. 
Supports integration with Weights & Biases (wandb), Loguru, TensorBoard, and other 
logging frameworks as needed.

This setup ensures consistent logging across various platforms, facilitating 
effective monitoring and debugging.

Example:
    from tools.logger import custom_logger
    custom_logger()
"""

import sys
from typing import List

from loguru import logger
from rich.console import Console
from rich.table import Table

from yolo.config.config import YOLOLayer


def custom_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def log_model(model: List[YOLOLayer]):
    console = Console()
    table = Table(title="Model Layers")

    table.add_column("Index", justify="center")
    table.add_column("Layer Type", justify="center")
    table.add_column("Tags", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Channels (IN->OUT)", justify="center")

    for idx, layer in enumerate(model, start=1):
        layer_param = sum(x.numel() for x in layer.parameters())  # number parameters
        in_channels, out_channels = getattr(layer, "in_c", None), getattr(layer, "out_c", None)
        if in_channels and out_channels:
            channels = f"{in_channels:4} -> {out_channels:4}"
        else:
            channels = "-"
        table.add_row(str(idx), layer.layer_type, layer.tags, f"{layer_param:,}", channels)
    console.print(table)
