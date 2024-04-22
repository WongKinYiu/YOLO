import argparse
from loguru import logger
from model.yolo import get_model
from utils.tools import load_model_cfg, custom_logger
import hydra
from config.config import Config


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    model = get_model(cfg.model)


if __name__ == "__main__":
    custom_logger()
    main()
