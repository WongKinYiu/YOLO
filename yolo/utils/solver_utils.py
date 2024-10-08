import torch

import numpy as np
from rich.table import Table


def format_prediction(prediction: torch.Tensor) -> dict:
    """
    Format the prediction output to a dictionary.
    Args:
        prediction (Tensor): A list of predictions.
        
    Returns:
        dict: A dictionary containing the formatted predictions.
              With the following keys:
                - "boxes": The bounding boxes in the format [x_min, y_min, x_max, y_max].
                - "scores": The confidence scores of the predictions.
                - "labels": The class IDs of the predictions.
    """
    return {
        "boxes": prediction[:, 1:5],
        "scores": prediction[:, 5],
        "labels": prediction[:, 0].int(),
    }

def format_target(target: torch.Tensor) -> dict:
    """
    Format the target output to a dictionary.
    Args:
        target (Tensor): A list of targets.
    
    Returns:
        dict: A dictionary containing the formatted targets.
              With the following keys:
                - "boxes": The bounding boxes in the format [x_min, y_min, x_max, y_max].
                - "labels": The class IDs of the targets.
    """    
    is_present = target[:, 0] != -1
    return {
        "boxes": target[is_present, 1:5],
        "labels": target[is_present, 0].int(),
    }