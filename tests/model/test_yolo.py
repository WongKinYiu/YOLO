import pytest
import torch
import sys

sys.path.append("./")
from utils.tools import load_model_cfg
from model.yolo import YOLO


def test_load_model_configuration():
    config_name = "v7-base"
    model_cfg = load_model_cfg(config_name)

    assert "model" in model_cfg
    assert isinstance(model_cfg, dict)


def test_build_model():
    config_name = "v7-base"
    model_cfg = load_model_cfg(config_name)
    model = YOLO(model_cfg)
    model.build_model(model_cfg["model"])
    assert len(model.model) == 106


def test_yolo_forward_output_shape():
    config_name = "v7-base"
    model_cfg = load_model_cfg(config_name)
    model = YOLO(model_cfg)

    # 2 - batch size, 3 - number of channels, 640x640 - image dimensions
    dummy_input = torch.rand(2, 3, 640, 640)

    # Forward pass through the model
    output = model(dummy_input)
    output_shape = [x.shape for x in output]
    assert output_shape == [
        torch.Size([2, 3, 20, 20, 85]),
        torch.Size([2, 3, 80, 80, 85]),
        torch.Size([2, 3, 40, 40, 85]),
    ]
