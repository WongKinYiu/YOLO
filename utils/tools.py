import os
import yaml
from loguru import logger
from typing import Dict, Any


def complete_path(file_name: str = "v7-base.yaml") -> str:
    """
    Ensures the path to a model configuration is a existing file

    Parameters:
        file_name (str): The filename or path, with default 'v7-base.yaml'.

    Returns:
        str: A complete path with necessary prefix and extension.
    """
    # Ensure the file has the '.yaml' extension if missing
    if not file_name.endswith(".yaml"):
        file_name += ".yaml"

    # Add folder prefix if only the filename is provided
    if os.path.dirname(file_name) == "":
        file_name = os.path.join("./config/model", file_name)

    return file_name


def load_model_cfg(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML configuration file, ensure necessary keys are present, and return its content as a dictionary.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file cannot be found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    file_path = complete_path(file_path)
    try:
        with open(file_path, "r") as file:
            model_cfg = yaml.safe_load(file) or {}

        # Check for required keys and set defaults if not present
        if "nc" not in model_cfg:
            model_cfg["nc"] = 80
            logger.warning("'nc' not found in the YAML file. Setting default 'nc' to 80.")

        if "anchor" not in model_cfg:
            logger.error("'anchor' is missing in the configuration file.")
            raise ValueError("Missing required key: 'anchor'")

        if "model" not in model_cfg:
            logger.error("'model' is missing in the configuration file.")
            raise ValueError("Missing required key: 'model'")

        return model_cfg

    except FileNotFoundError:
        logger.error(f"YAML file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
