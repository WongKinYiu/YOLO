import sys
from math import isclose
from pathlib import Path

import pytest
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import StreamDataLoader
from yolo.tools.solver import TrainModel, ValidateModel
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box


@pytest.fixture
def model_validator(validation_cfg: Config):
    validator = ValidateModel(validation_cfg)
    return validator


def test_model_validator_initialization(solver: Trainer, model_validator: ValidateModel):
    assert isinstance(model_validator.model, YOLO)
    assert hasattr(solver, "validate")


def test_model_validator_solve_mock_dataset(
    solver: Trainer, model_validator: ValidateModel, validation_dataloader: DataLoader
):
    mAPs = solver.validate(model_validator, dataloaders=validation_dataloader)[0]
    except_mAPs = {"map_50": 0.7379, "map": 0.5617}
    assert isclose(mAPs["map_50"], except_mAPs["map_50"], abs_tol=1e-4)
    assert isclose(mAPs["map"], except_mAPs["map"], abs_tol=1e-4)


@pytest.fixture
def model_tester(inference_cfg: Config, model: YOLO, vec2box: Vec2Box, validation_progress_logger, device):
    tester = ModelTester(inference_cfg, model, vec2box, validation_progress_logger, device)
    return tester


@pytest.fixture
def modelv7_tester(inference_v7_cfg: Config, model_v7: YOLO, anc2box: Anc2Box, validation_progress_logger, device):
    tester = ModelTester(inference_v7_cfg, model_v7, anc2box, validation_progress_logger, device)
    return tester


def test_model_tester_initialization(model_tester: ModelTester):
    assert isinstance(model_tester.model, YOLO)
    assert hasattr(model_tester, "solve")


def test_model_tester_solve_single_image(model_tester: ModelTester, file_stream_data_loader: StreamDataLoader):
    model_tester.solve(file_stream_data_loader)


def test_modelv7_tester_solve_single_image(modelv7_tester: ModelTester, file_stream_data_loader_v7: StreamDataLoader):
    modelv7_tester.solve(file_stream_data_loader_v7)


@pytest.fixture
def model_trainer(train_cfg: Config, model: YOLO, vec2box: Vec2Box, train_progress_logger, device):
    train_cfg.task.epoch = 2
    trainer = TrainModel(train_cfg)
    return trainer


def test_model_trainer_initialization(solver: Trainer, model_trainer: TrainModel):
    assert isinstance(model_trainer.model, YOLO)
    assert hasattr(solver, "fit")
    assert solver.optimizers is not None


# def test_model_trainer_solve_mock_dataset(model_trainer: ModelTrainer, train_dataloader: YoloDataLoader):
#     model_trainer.solve(train_dataloader)
