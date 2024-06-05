import os
import zipfile
from typing import Optional

import requests
from loguru import logger
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from yolo.config.config import DatasetConfig


def download_file(url, destination):
    """
    Downloads a file from the specified URL to the destination path with progress logging.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            "{task.completed}/{task.total} bytes",
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"📥 Downloading {os.path.basename(destination)}...", total=total_size)
            with open(destination, "wb") as file:
                for data in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    file.write(data)
                    progress.update(task, advance=len(data))
    logger.info("✅ Download completed.")


def unzip_file(source, destination):
    """
    Extracts a ZIP file to the specified directory and removes the ZIP file after extraction.
    """
    logger.info(f"Unzipping {os.path.basename(source)}...")
    with zipfile.ZipFile(source, "r") as zip_ref:
        zip_ref.extractall(destination)
    os.remove(source)
    logger.info(f"Removed {source}.")


def check_files(directory, expected_count=None):
    """
    Returns True if the number of files in the directory matches expected_count, False otherwise.
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files) == expected_count if expected_count is not None else bool(files)


def prepare_dataset(cfg: DatasetConfig):
    """
    Prepares dataset by downloading and unzipping if necessary.
    """
    data_dir = cfg.path
    for data_type, settings in cfg.auto_download.items():
        base_url = settings["base_url"]
        for dataset_type, dataset_args in settings.items():
            if dataset_type == "base_url":
                continue  # Skip the base_url entry
            file_name = f"{dataset_args.get('file_name', dataset_type)}.zip"
            url = f"{base_url}{file_name}"
            local_zip_path = os.path.join(data_dir, file_name)
            extract_to = os.path.join(data_dir, data_type) if data_type != "annotations" else data_dir
            final_place = os.path.join(extract_to, dataset_type)

            os.makedirs(final_place, exist_ok=True)
            if check_files(final_place, dataset_args.get("file_num")):
                logger.info(f"✅ Dataset {dataset_type: <12} already verified.")
                continue

            if not os.path.exists(local_zip_path):
                download_file(url, local_zip_path)
            unzip_file(local_zip_path, extract_to)

            if not check_files(final_place, dataset_args.get("file_num")):
                logger.error(f"Error verifying the {dataset_type} dataset after extraction.")


def prepare_weight(downlaod_link: Optional[str] = None, weight_path: str = "v9-c.pt"):
    weight_name = os.path.basename(weight_path)
    if downlaod_link is None:
        downlaod_link = "https://github.com/WongKinYiu/yolov9mit/releases/download/v1.0-alpha/"
    weight_link = f"{downlaod_link}{weight_name}"

    if not os.path.isdir(os.path.dirname(weight_path)):
        os.makedirs(os.path.dirname(weight_path))

    if os.path.exists(weight_path):
        logger.info(f"Weight file '{weight_path}' already exists.")
    try:
        download_file(weight_link, weight_path)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download the weight file: {e}")


if __name__ == "__main__":
    import sys

    sys.path.append("./")
    from utils.logging_utils import custom_logger

    custom_logger()
    prepare_weight()
