import json
from pathlib import Path
from typing import Dict, List, Optional

from rich.progress import track


def discretize_categories(categories: List[Dict[str, int]]) -> Dict[int, int]:
    """
    Maps each unique 'id' in the list of category dictionaries to a sequential integer index.
    Indices are assigned based on the sorted 'id' values.
    """
    sorted_categories = sorted(categories, key=lambda category: category["id"])
    return {category["id"]: index for index, category in enumerate(sorted_categories)}


def process_annotations(
    image_annotations: Dict[int, List[Dict]],
    image_info_dict: Dict[int, tuple],
    output_dir: Path,
    id_to_idx: Optional[Dict[int, int]] = None,
) -> None:
    """
    Process and save annotations to files, with option to remap category IDs.
    """
    for image_id, annotations in track(image_annotations.items(), description="Processing annotations"):
        file_path = output_dir / "{image_id:0>12}.txt"
        if not annotations:
            continue
        with open(file_path, "w") as file:
            for annotation in annotations:
                process_annotation(annotation, image_info_dict[image_id], id_to_idx, file)


def process_annotation(annotation: Dict, image_dims: tuple, id_to_idx: Optional[Dict[int, int]], file) -> None:
    """
    Convert a single annotation's segmentation and write it to the open file handle.
    """
    category_id = annotation["category_id"]
    segmentation = (
        annotation["segmentation"][0]
        if annotation["segmentation"] and isinstance(annotation["segmentation"][0], list)
        else None
    )

    if segmentation is None:
        return

    img_width, img_height = image_dims
    normalized_segmentation = normalize_segmentation(segmentation, img_width, img_height)

    if id_to_idx:
        category_id = id_to_idx.get(category_id, category_id)

    file.write(f"{category_id} {' '.join(normalized_segmentation)}\n")


def normalize_segmentation(segmentation: List[float], img_width: int, img_height: int) -> List[str]:
    """
    Normalize and format segmentation coordinates.
    """
    normalized = [
        f"{coord / img_width:.6f}" if index % 2 == 0 else f"{coord / img_height:.6f}"
        for index, coord in enumerate(segmentation)
    ]
    return normalized


def convert_annotations(json_file: str, output_dir: str) -> None:
    """
    Load annotation data from a JSON file and process all annotations.
    """
    with open(json_file) as file:
        data = json.load(file)

    Path(output_dir).mkdir(exist_ok=True)

    image_info_dict = {img["id"]: (img["width"], img["height"]) for img in data.get("images", [])}
    id_to_idx = discretize_categories(data.get("categories", [])) if "categories" in data else None
    image_annotations = {img_id: [] for img_id in image_info_dict}

    for annotation in data.get("annotations", []):
        if not annotation.get("iscrowd", False):
            image_annotations[annotation["image_id"]].append(annotation)

    process_annotations(image_annotations, image_info_dict, output_dir, id_to_idx)


if __name__ == "__main__":
    convert_annotations("./data/coco/annotations/instances_train2017.json", "./data/coco/labels/train2017/")
    convert_annotations("./data/coco/annotations/instances_val2017.json", "./data/coco/labels/val2017/")
