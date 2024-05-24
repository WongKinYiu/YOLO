import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.model.yolo import YOLO, get_model

config_path = "../../yolo/config/model"
config_name = "v7-base"


def test_build_model():
    with initialize(config_path=config_path, version_base=None):
        model_cfg = compose(config_name=config_name)
        OmegaConf.set_struct(model_cfg, False)
        model = YOLO(model_cfg)
        model.build_model(model_cfg.model)
        assert len(model.model) == 106


def test_get_model():
    with initialize(config_path=config_path, version_base=None):
        model_cfg = compose(config_name=config_name)
        model = get_model(model_cfg)
        assert isinstance(model, YOLO)


def test_yolo_forward_output_shape():
    with initialize(config_path=config_path, version_base=None):
        model_cfg = compose(config_name=config_name)

        model = get_model(model_cfg)
        # 2 - batch size, 3 - number of channels, 640x640 - image dimensions
        dummy_input = torch.rand(2, 3, 640, 640)

        # Forward pass through the model
        output = model(dummy_input)
        output_shape = [x.shape for x in output[-1]]
        assert output_shape == [
            torch.Size([2, 3, 20, 20, 85]),
            torch.Size([2, 3, 80, 80, 85]),
            torch.Size([2, 3, 40, 40, 85]),
        ]
