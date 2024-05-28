import sys
from pathlib import Path

import hydra
import torch
from loguru import logger

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import get_model
from yolo.tools.log_helper import custom_logger
from yolo.tools.trainer import Trainer
from yolo.utils.dataloader import get_dataloader
from yolo.utils.drawer import draw_model
from yolo.utils.get_dataset import prepare_dataset


@hydra.main(config_path="../yolo/config", config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.download.auto:
        prepare_dataset(cfg.download)

    dataloader = get_dataloader(cfg)
    model = get_model(cfg)
    draw_model(model=model)
    # TODO: get_device or rank, for DDP mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(model, cfg, device)
    trainer.train(dataloader, 10)


if __name__ == "__main__":
    custom_logger()
    main()
