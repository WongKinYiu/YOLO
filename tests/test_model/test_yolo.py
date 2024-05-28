import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.model.yolo import YOLO, get_model

config_path = "../../yolo/config"
config_name = "config"


def test_build_model():
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)

        OmegaConf.set_struct(cfg.model, False)
        model = YOLO(cfg.model, 80)
        assert len(model.model) == 38


def test_get_model():
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
        model = get_model(cfg)
        assert isinstance(model, YOLO)


def test_yolo_forward_output_shape():
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
        model = get_model(cfg)
        # 2 - batch size, 3 - number of channels, 640x640 - image dimensions
        dummy_input = torch.rand(2, 3, 640, 640)

        # Forward pass through the model
        output = model(dummy_input)
        output_shape = [x.shape for x in output[-1]]
        assert output_shape == [
            torch.Size([2, 144, 80, 80]),
            torch.Size([2, 144, 40, 40]),
            torch.Size([2, 144, 20, 20]),
            torch.Size([2, 144, 80, 80]),
            torch.Size([2, 144, 40, 40]),
            torch.Size([2, 144, 20, 20]),
        ]
