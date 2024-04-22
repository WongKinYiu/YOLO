from loguru import logger
from model.yolo import get_model
from tools.log_helper import custom_logger
from utils.get_dataset import prepare_dataset
import hydra
from config.config import Config


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.download.auto:
        prepare_dataset(cfg.download)

    model = get_model(cfg.model)


if __name__ == "__main__":
    custom_logger()
    main()
