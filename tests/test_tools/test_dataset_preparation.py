import os
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.dataset_preparation import prepare_dataset, prepare_weight


def test_prepare_dataset(train_cfg: Config):
    dataset_path = Path("tests/data")
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    prepare_dataset(train_cfg.dataset, task="train")
    prepare_dataset(train_cfg.dataset, task="val")

    images_path = Path("tests/data/images")
    for data_type in images_path.iterdir():
        assert len(os.listdir(data_type)) == 5

    annotations_path = Path("tests/data/annotations")
    assert "instances_val.json" in os.listdir(annotations_path)
    assert "instances_train.json" in os.listdir(annotations_path)


def test_prepare_weight():
    prepare_weight()
