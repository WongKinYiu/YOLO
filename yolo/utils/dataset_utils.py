import json
import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from yolo.tools.data_conversion import discretize_categories


def locate_label_paths(dataset_path: Path, phase_name: Path) -> Tuple[Path, Path]:
    """
    Find the path to label files for a specified dataset and phase(e.g. training).

    Args:
        dataset_path (Path): The path to the root directory of the dataset.
        phase_name (Path): The name of the phase for which labels are being searched (e.g., "train", "val", "test").

    Returns:
        Tuple[Path, Path]: A tuple containing the path to the labels file and the file format ("json" or "txt").
    """
    json_labels_path = dataset_path / "annotations" / f"instances_{phase_name}.json"

    txt_labels_path = dataset_path / "labels" / phase_name

    if json_labels_path.is_file():
        return json_labels_path, "json"

    elif txt_labels_path.is_dir():
        txt_files = [f for f in os.listdir(txt_labels_path) if f.endswith(".txt")]
        if txt_files:
            return txt_labels_path, "txt"

    logger.warning("No labels found in the specified dataset path and phase name.")
    return [], None


def create_image_metadata(
        labels_path: str
) -> Tuple[Dict[int, List], Dict[int, Dict], Dict[str, int]]:
    """
    Returnes three dictionaries mapping image id to list of annotations, 
    image id to image information, and image name to image id.
    Image id is the `int` `id` assigned to a image in the COCO formatted .json file.

    Args:
        labels_path (str): The path to the annotation json file.

    Returns:
        (annotations_dict, image_info_dict, image_name_to_id_dict):
            annotations_dict is a dictionary where keys are image ids and values
            are lists of annotation dictionaries.
            image_info_dict is a dictionary where keys are image file id and
            values are image information dictionaries.
            image_name_to_id_dict is a dictionary with image name without
            extension as key and int image id as value.
    """
    with open(labels_path, "r") as file:
        json_data = json.load(file)
        image_name_to_id_dict = {
            Path(img["file_name"]).name: img['id'] for img in json_data["images"]
        }
        id_to_idx = discretize_categories(json_data.get("categories", [])) if "categories" in json_data else None
        annotations_dict = organize_annotations_by_image(json_data, id_to_idx)  # check lookup is a good name?
        image_info_dict = {img['id']: img for img in json_data["images"]}
        return annotations_dict, image_info_dict, image_name_to_id_dict


def organize_annotations_by_image(
        json_data: Dict[str, Any],
        category_id_to_idx: Optional[Dict[int, int]],
) -> dict[int, list[dict]]:
    """
    Returns a dict mapping image id to a list of all corresponding annotations.

    Annotations with "iscrowd" set to True, are excluded. Image id is the `int`
    `image_id` in the corresponding annotation dict stored in the
    COCO formatted .json file.

    Args:
        json_data: Data read from a COCO json file.
        category_id_to_idx: For COCO dataset, a dict mapping from category_id
            to (category_id - 1).
    Returns:
        image_name_to_annotation_dict_list: A dictionary where keys are image ids
            and values are lists of annotation dictionaries.
    """
    image_id_to_annotation_dict_list = {}
    for annotation_dict in json_data["annotations"]:
        if annotation_dict["iscrowd"]:
            continue
        image_id = annotation_dict["image_id"]
        if category_id_to_idx:
            annotation_dict["category_id"] = category_id_to_idx[annotation_dict["category_id"]]
        if image_id not in image_id_to_annotation_dict_list:
            image_id_to_annotation_dict_list[image_id] = []
        image_id_to_annotation_dict_list[image_id].append(annotation_dict)
    return image_id_to_annotation_dict_list


def scale_segmentation(
    annotations: List[Dict[str, Any]], image_dimensions: Dict[str, int]
) -> Optional[List[List[float]]]:
    """
    Scale the segmentation data based on image dimensions and return a list of scaled segmentation data.

    Args:
        annotations (List[Dict[str, Any]]): A list of annotation dictionaries.
        image_dimensions (Dict[str, int]): A dictionary containing image dimensions (height and width).

    Returns:
        Optional[List[List[float]]]: A list of scaled segmentation data, where each sublist contains category_id followed by scaled (x, y) coordinates.
    """
    if annotations is None:
        return None

    seg_array_with_cat = []
    h, w = image_dimensions["height"], image_dimensions["width"]
    for anno in annotations:
        category_id = anno["category_id"]
        if "segmentation" in anno:
            seg_list = [item for sublist in anno["segmentation"] for item in sublist]
        elif "bbox" in anno:
            seg_list = anno["bbox"]
        scaled_seg_data = (
            np.array(seg_list).reshape(-1, 2) / [w, h]
        ).tolist()  # make the list group in x, y pairs and scaled with image width, height
        scaled_flat_seg_data = [category_id] + list(chain(*scaled_seg_data))  # flatten the scaled_seg_data list
        seg_array_with_cat.append(scaled_flat_seg_data)

    return seg_array_with_cat
