import argparse
from loguru import logger
from model.yolo import get_model
from utils.tools import load_model_cfg, custom_logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to get the model configuration file.

    Returns:
        argparse.Namespace: The command-line arguments object with 'config' attribute.
    """
    parser = argparse.ArgumentParser(description="Load a YOLO model configuration and display the model.")
    parser.add_argument(
        "--model-config", type=str, default="v7-base", help="Name or path to the model configuration file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    custom_logger()
    args = parse_arguments()
    model_cfg = load_model_cfg(args.model_config)
    model = get_model(model_cfg)
    logger.info("Success load model")
