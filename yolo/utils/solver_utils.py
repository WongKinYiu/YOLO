import contextlib
import io

import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rich.table import Table


def calculate_ap(coco_gt: COCO, pd_path):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(pd_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval.stats


def make_ap_table(score, past_result=[], last_score=None, epoch=-1):
    ap_table = Table()
    ap_table.add_column("Epoch", justify="center", style="white", width=5)
    ap_table.add_column("Avg. Precision", justify="left", style="cyan")
    ap_table.add_column("%", justify="right", style="green", width=5)
    ap_table.add_column("Avg. Recall", justify="left", style="cyan")
    ap_table.add_column("%", justify="right", style="green", width=5)

    for eps, (ap_name1, ap_color1, ap_value1, ap_name2, ap_color2, ap_value2) in past_result:
        ap_table.add_row(f"{eps: 3d}", ap_name1, f"{ap_color1}{ap_value1:.2f}", ap_name2, f"{ap_color2}{ap_value2:.2f}")
    if past_result:
        ap_table.add_row()

    color = np.where(last_score <= score, "[green]", "[red]")

    this_ap = ("AP @ .5:.95", color[0], score[0], "AP @        .5", color[1], score[1])
    metrics = [
        ("AP @ .5:.95", color[0], score[0], "AR maxDets   1", color[6], score[6]),
        ("AP @     .5", color[1], score[1], "AR maxDets  10", color[7], score[7]),
        ("AP @    .75", color[2], score[2], "AR maxDets 100", color[8], score[8]),
        ("AP  (small)", color[3], score[3], "AR     (small)", color[9], score[9]),
        ("AP (medium)", color[4], score[4], "AR    (medium)", color[10], score[10]),
        ("AP  (large)", color[5], score[5], "AR     (large)", color[11], score[11]),
    ]

    for ap_name, ap_color, ap_value, ar_name, ar_color, ar_value in metrics:
        ap_table.add_row(f"{epoch: 3d}", ap_name, f"{ap_color}{ap_value:.2f}", ar_name, f"{ar_color}{ar_value:.2f}")

    return ap_table, this_ap

def merge_coco_objects(coco_list):
    """
    Merge multiple COCO objects into a single one.

    Args:
        coco_list (List[COCO]): A list of COCO objects to merge.

    Returns:
        COCO: The merged COCO object.
    """
    if not coco_list:
        return None  # Return None if the list is empty

    if len(coco_list) == 1:
        return coco_list[0]  # If there's only one object, return it directly

    # Start with the first COCO object
    merged_coco = COCO()
    merged_coco.dataset = copy.deepcopy(coco_list[0].dataset)

    for coco in coco_list[1:]:
        # Check if the data is identical to the already merged data
        if coco.dataset == merged_coco.dataset:
            continue  # If identical, skip this object

        # Merge images
        max_img_id = max(merged_coco.imgs.keys()) if merged_coco.imgs else 0
        for img_id, img_info in coco.imgs.items():
            if img_info not in merged_coco.dataset['images']:
                new_img_id = img_id + max_img_id
                merged_coco.dataset['images'].append({**img_info, 'id': new_img_id})

        # Merge categories
        max_cat_id = max(merged_coco.cats.keys()) if merged_coco.cats else 0
        for cat_id, cat_info in coco.cats.items():
            if cat_info not in merged_coco.dataset['categories']:
                new_cat_id = cat_id + max_cat_id
                merged_coco.dataset['categories'].append({**cat_info, 'id': new_cat_id})

        # Merge annotations
        max_ann_id = max(ann['id'] for ann in merged_coco.dataset['annotations']) if merged_coco.dataset['annotations'] else 0
        for ann in coco.dataset['annotations']:
            if ann not in merged_coco.dataset['annotations']:
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = ann['id'] + max_ann_id
                new_ann['image_id'] = ann['image_id'] + max_img_id
                new_ann['category_id'] = ann['category_id'] + max_cat_id
                merged_coco.dataset['annotations'].append(new_ann)

    # Create index
    merged_coco.createIndex()

    return merged_coco