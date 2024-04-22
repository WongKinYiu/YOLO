import os
import zipfile

import hydra
from loguru import logger
import requests
from tqdm.rich import tqdm


def download_file(url, dest_path):
    """
    Downloads a file from a specified URL to a destination path with progress logging.
    """
    logger.info(f"Downloading {os.path.basename(dest_path)}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total_length, unit="iB", unit_scale=True, desc=os.path.basename(dest_path), leave=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    logger.info("Download complete!")


def unzip_file(zip_path, extract_to):
    """
    Unzips a ZIP file to a specified directory.
    """
    logger.info(f"Unzipping {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
    logger.info(f"Removed {zip_path}")


def check_files(directory, expected_count):
    """
    Checks if the specified directory has the expected number of files.
    """
    num_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    return num_files == expected_count


@hydra.main(config_path="../config/data", config_name="download", version_base=None)
def prepare_dataset(download_cfg):
    data_dir = download_cfg.path
    base_url = download_cfg.images.base_url
    datasets = download_cfg.images.datasets

    for dataset_type in datasets:
        file_name, expected_files = datasets[dataset_type].values()
        url = f"{base_url}{file_name}"
        local_zip_path = os.path.join(data_dir, file_name)
        extract_to = os.path.join(data_dir, dataset_type, "images")

        # Ensure the extraction directory exists
        os.makedirs(extract_to, exist_ok=True)

        # Check if the correct number of files exists
        if check_files(extract_to, expected_files):
            logger.info(f"Dataset {dataset_type} already verified.")
            continue

        if os.path.exists(local_zip_path):
            logger.info(f"Dataset {dataset_type} already downloaded.")
        else:
            download_file(url, local_zip_path)

        unzip_file(local_zip_path, extract_to)

        print(os.path.exists(local_zip_path), check_files(extract_to, expected_files))

        # Additional verification post extraction
        if not check_files(extract_to, expected_files):
            logger.error(f"Error in verifying the {dataset_type} dataset after extraction.")


if __name__ == "__main__":
    from tools.log_helper import custom_logger

    custom_logger()
    download_coco_dataset()
