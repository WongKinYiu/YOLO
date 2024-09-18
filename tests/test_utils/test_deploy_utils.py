import pytest
import torch

from yolo.config.config import Config
from yolo.utils.deploy_utils import FastModelLoader


def test_load_v7_onnx(inference_v7_onnx_cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastModelLoader(inference_v7_onnx_cfg).load_model(device)
    assert hasattr(model, "num_classes")


def test_infer_v7_onnx(inference_v7_onnx_cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastModelLoader(inference_v7_onnx_cfg).load_model(device)
    image_data = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    predict = model(image_data)
    assert "Main" in predict
    predictions = predict["Main"]
    assert len(predictions) == 3
    assert predictions[0].shape == (1, 255, 80, 80)
    assert predictions[1].shape == (1, 255, 40, 40)
    assert predictions[2].shape == (1, 255, 20, 20)
