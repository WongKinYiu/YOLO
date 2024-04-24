import hydra
from loguru import logger

from config.config import Config
from model.yolo import get_model
from tools.log_helper import custom_logger
from utils.dataloader import YoloDataset
from utils.get_dataset import prepare_dataset


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    dataset = YoloDataset(cfg)
    if cfg.download.auto:
        prepare_dataset(cfg.download)

    model = get_model(cfg.model)


if __name__ == "__main__":
    custom_logger()
    main()
