import hydra
import torch
from loguru import logger

from config.config import Config
from model.yolo import get_model
from tools.log_helper import custom_logger
from tools.trainer import Trainer
from utils.dataloader import get_dataloader
from utils.get_dataset import prepare_dataset


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.download.auto:
        prepare_dataset(cfg.download)

    dataloader = get_dataloader(cfg)
    model = get_model(cfg.model)
    # TODO: get_device or rank, for DDP mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(model, cfg.hyper.train, device)
    trainer.train(dataloader, 10)


if __name__ == "__main__":
    custom_logger()
    main()
