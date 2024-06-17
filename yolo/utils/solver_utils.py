import os
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rich.table import Table


def calculate_ap(coco_gt: COCO, pd_path):
    sys.stdout = open(os.devnull, "w")
    coco_dt = coco_gt.loadRes(pd_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    sys.stdout = sys.__stdout__
    return coco_eval.stats


def make_ap_table(score, past_result=[], epoch=-1):
    ap_table = Table()
    ap_table.add_column("Epoch", justify="center", style="white", width=5)
    ap_table.add_column("Avg. Precision", justify="left", style="cyan")
    ap_table.add_column("", justify="right", style="green", width=5)
    ap_table.add_column("Avg. Recall", justify="left", style="cyan")
    ap_table.add_column("", justify="right", style="green", width=5)

    for eps, (ap_name1, ap_value1, ap_name2, ap_value2) in past_result:
        ap_table.add_row(f"{eps: 3d}", ap_name1, f"{ap_value1:.2f}", ap_name2, f"{ap_value2:.2f}")
    if past_result:
        ap_table.add_row()

    this_ap = ("AP @ .5:.95", score[0], "AP @        .5", score[1])
    metrics = [
        ("AP @ .5:.95", score[0], "AR maxDets   1", score[6]),
        ("AP @     .5", score[1], "AR maxDets  10", score[7]),
        ("AP @    .75", score[2], "AR maxDets 100", score[8]),
        ("AP  (small)", score[3], "AR     (small)", score[9]),
        ("AP (medium)", score[4], "AR    (medium)", score[10]),
        ("AP  (large)", score[5], "AR     (large)", score[11]),
    ]

    for ap_name, ap_value, ar_name, ar_value in metrics:
        ap_table.add_row(f"{epoch: 3d}", ap_name, f"{ap_value:.2f}", ar_name, f"{ar_value:.2f}")

    return ap_table, this_ap
