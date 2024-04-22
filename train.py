import argparse
from loguru import logger
from model.yolo import get_model
from utils.tools import load_model_cfg, custom_logger
from utils.get_dataset import download_coco_dataset
import hydra
from config.config import Config
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    OmegaConf.set_struct(cfg, False)
    download_coco_dataset(cfg.download)
    model = get_model(cfg.model)
    logger.info("Success load model")


if __name__ == "__main__":
    custom_logger()
    main()
