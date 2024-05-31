import sys
from pathlib import Path

import hydra
import torch
from loguru import logger

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.log_helper import custom_logger, get_valid_folder
from yolo.tools.trainer import Trainer
from yolo.utils.dataloader import get_dataloader
from yolo.utils.get_dataset import prepare_dataset


@hydra.main(config_path="../yolo/config", config_name="config", version_base=None)
def main(cfg: Config):
    custom_logger()
    save_path = get_valid_folder(cfg.hyper.general, cfg.name)
    if cfg.download.auto:
        prepare_dataset(cfg.download)

    dataloader = get_dataloader(cfg)
    # TODO: get_device or rank, for DDP mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, save_path, device)
    trainer.train(dataloader, cfg.hyper.train.epoch)


if __name__ == "__main__":
    main()
