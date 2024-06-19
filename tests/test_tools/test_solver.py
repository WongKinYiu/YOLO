import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import (
    Config,
    DataConfig,
    LossConfig,
    TrainConfig,
    ValidationConfig,
)
from yolo.model.yolo import YOLO, create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.loss_functions import create_loss_function
from yolo.tools.solver import (  # Adjust the import to your module
    ModelTester,
    ModelTrainer,
    ModelValidator,
)
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.logging_utils import ProgressLogger
from yolo.utils.model_utils import (
    ExponentialMovingAverage,
    create_optimizer,
    create_scheduler,
)


@pytest.fixture
def cfg() -> Config:
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg: Config = compose(config_name="config")
        cfg.weight = None
    return cfg


@pytest.fixture
def cfg_validaion() -> Config:
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg: Config = compose(config_name="config", overrides=["task=validation"])
        cfg.weight = None
    return cfg


@pytest.fixture
def cfg_inference() -> Config:
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg: Config = compose(config_name="config", overrides=["task=inference"])
        cfg.weight = None
    return cfg


@pytest.fixture
def device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


@pytest.fixture
def model(cfg: Config, device) -> YOLO:
    model = create_model(cfg.model, weight_path=None)
    return model.to(device)


@pytest.fixture
def vec2box(cfg: Config, model: YOLO, device) -> Vec2Box:
    model = create_model(cfg.model, weight_path=None).to(device)
    vec2box = Vec2Box(model, cfg.image_size, device)
    return vec2box


@pytest.fixture
def progress_logger(cfg: Config):
    progress_logger = ProgressLogger(cfg, exp_name=cfg.name)
    return progress_logger


def test_model_trainer_initialization(cfg: Config, model: YOLO, vec2box: Vec2Box, progress_logger, device):
    trainer = ModelTrainer(cfg, model, vec2box, progress_logger, device, use_ddp=False)
    assert trainer.model == model
    assert trainer.device == device
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.loss_fn is not None
    assert trainer.progress == progress_logger


# def test_model_trainer_train_one_batch(config, model, vec2box, progress_logger, device):
#     trainer = ModelTrainer(config, model, vec2box, progress_logger, device, use_ddp=False)
#     images = torch.rand(1, 3, 224, 224)
#     targets = torch.rand(1, 5)
#     loss_item = trainer.train_one_batch(images, targets)
#     assert isinstance(loss_item, dict)


def test_model_validator_initialization(cfg_validaion: Config, model: YOLO, vec2box: Vec2Box, progress_logger, device):
    validator = ModelValidator(cfg_validaion.task, model, vec2box, progress_logger, device)
    assert validator.model == model
    assert validator.device == device
    assert validator.progress == progress_logger


def test_model_tester_initialization(cfg_inference: Config, model: YOLO, vec2box: Vec2Box, progress_logger, device):
    tester = ModelTester(cfg_inference, model, vec2box, progress_logger, device)
    assert tester.model == model
    assert tester.device == device
    assert tester.progress == progress_logger
