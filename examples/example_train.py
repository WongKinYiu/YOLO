import sys
from pathlib import Path

import hydra
import torch
from loguru import logger

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.data_loader import create_dataloader
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.tools.trainer import ModelTrainer
from yolo.utils.logging_utils import custom_logger, validate_log_directory


@hydra.main(config_path="../yolo/config", config_name="config", version_base=None)
def main(cfg: Config):
    custom_logger()
    save_path = validate_log_directory(cfg.hyper.general, cfg.name)
    if cfg.download.auto:
        prepare_dataset(cfg.download)

    dataloader = create_dataloader(cfg)
    # TODO: get_device or rank, for DDP mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ModelTrainer(cfg, save_path, device)
    trainer.train(dataloader, cfg.hyper.train.epoch)


if __name__ == "__main__":
    main()
