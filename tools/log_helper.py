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
from loguru import logger


def custom_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
