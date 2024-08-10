import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import Tensor, arange, tensor
from torchvision.ops import batched_nms

from yolo.config.config import AnchorConfig, MatcherConfig, ModelConfig, NMSConfig
from yolo.model.yolo import YOLO


def calculate_iou(bbox1, bbox2, metrics="iou") -> Tensor:
    metrics = metrics.lower()
    EPS = 1e-9
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    # Expand dimensions if necessary
    if bbox1.ndim == 2 and bbox2.ndim == 2:
        bbox1 = bbox1.unsqueeze(1)  # (Ax4) -> (Ax1x4)
        bbox2 = bbox2.unsqueeze(0)  # (Bx4) -> (1xBx4)
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZxAx4) -> (BZxAx1x4)
        bbox2 = bbox2.unsqueeze(1)  # (BZxBx4) -> (BZx1xBx4)

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)

    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + EPS)
    if metrics == "iou":
        return iou.to(dtype)

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou.to(dtype)

    # Compute aspect ratio penalty term
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    alpha = v / (v - iou + 1 + EPS)
    # Compute CIoU
    ciou = diou - alpha * v
    return ciou.to(dtype)


def transform_bbox(bbox: Tensor, indicator="xywh -> xyxy"):
    data_type = bbox.dtype
    in_type, out_type = indicator.replace(" ", "").split("->")

    if in_type not in ["xyxy", "xywh", "xycwh"] or out_type not in ["xyxy", "xywh", "xycwh"]:
        raise ValueError("Invalid input or output format")

    if in_type == "xywh":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 0] + bbox[..., 2]
        y_max = bbox[..., 1] + bbox[..., 3]
    elif in_type == "xyxy":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 2]
        y_max = bbox[..., 3]
    elif in_type == "xycwh":
        x_min = bbox[..., 0] - bbox[..., 2] / 2
        y_min = bbox[..., 1] - bbox[..., 3] / 2
        x_max = bbox[..., 0] + bbox[..., 2] / 2
        y_max = bbox[..., 1] + bbox[..., 3] / 2

    if out_type == "xywh":
        bbox = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=-1)
    elif out_type == "xyxy":
        bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    elif out_type == "xycwh":
        bbox = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min], dim=-1)

    return bbox.to(dtype=data_type)


def generate_anchors(image_size: List[int], strides: List[int]):
    """
    Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, H, stride) + shift
        w = torch.arange(0, W, stride) + shift
        anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


class BoxMatcher:
    def __init__(self, cfg: MatcherConfig, class_num: int, anchors: Tensor) -> None:
        self.class_num = class_num
        self.anchors = anchors
        for attr_name in cfg:
            setattr(self, attr_name, cfg[attr_name])

    def get_valid_matrix(self, target_bbox: Tensor):
        """
        Get a boolean mask that indicates whether each target bounding box overlaps with each anchor.

        Args:
            target_bbox [batch x targets x 4]: The bounding box of each targets.
        Returns:
            [batch x targets x anchors]: A boolean tensor indicates if target bounding box overlaps with anchors.
        """
        Xmin, Ymin, Xmax, Ymax = target_bbox[:, :, None].unbind(3)
        anchors = self.anchors[None, None]  # add a axis at first, second dimension
        anchors_x, anchors_y = anchors.unbind(dim=3)
        target_in_x = (Xmin < anchors_x) & (anchors_x < Xmax)
        target_in_y = (Ymin < anchors_y) & (anchors_y < Ymax)
        target_on_anchor = target_in_x & target_in_y
        return target_on_anchor

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """
        Get the (predicted class' probabilities) corresponding to the target classes across all anchors

        Args:
            predict_cls [batch x anchors x class]: The predicted probabilities for each class across each anchor.
            target_cls [batch x targets]: The class index for each target.

        Returns:
            [batch x targets x anchors]: The probabilities from `pred_cls` corresponding to the class indices specified in `target_cls`.
        """
        predict_cls = predict_cls.transpose(1, 2)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))
        cls_probabilities = torch.gather(predict_cls, 1, target_cls)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox, target_bbox) -> Tensor:
        """
        Get the IoU between each target bounding box and each predicted bounding box.

        Args:
            predict_bbox [batch x predicts x 4]: Bounding box with [x1, y1, x2, y2].
            target_bbox [batch x targets x 4]: Bounding box with [x1, y1, x2, y2].
        Returns:
            [batch x targets x predicts]: The IoU scores between each target and predicted.
        """
        return calculate_iou(target_bbox, predict_bbox, self.iou).clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, topk: int = 10) -> Tuple[Tensor, Tensor]:
        """
        Filter the top-k suitability of targets for each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors
            topk (int, optional): Number of top scores to retain per anchor.

        Returns:
            topk_targets [batch x targets x anchors]: Only leave the topk targets for each anchor
            topk_masks [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.
        """
        values, indices = target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_masks = topk_targets > 0
        return topk_targets, topk_masks

    def filter_duplicates(self, target_matrix: Tensor):
        """
        Filter the maximum suitability target index of each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors

        Returns:
            unique_indices [batch x anchors x 1]: The index of the best targets for each anchors
        """
        # TODO: add a assert for no target on the image
        unique_indices = target_matrix.argmax(dim=1)
        return unique_indices[..., None]

    def __call__(self, target: Tensor, predict: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        1. For each anchor prediction, find the highest suitability targets
        2. Select the targets
        2. Noramlize the class probilities of targets
        """
        predict_cls, predict_bbox = predict
        target_cls, target_bbox = target.split([1, 4], dim=-1)  # B x N x (C B) -> B x N x C, B x N x B
        target_cls = target_cls.long().clamp(0)

        # get valid matrix (each gt appear in which anchor grid)
        grid_mask = self.get_valid_matrix(target_bbox)

        # get iou matrix (iou with each gt bbox and each predict anchor)
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)

        # get cls matrix (cls prob with each gt class and each predict class)
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)

        target_matrix = grid_mask * (iou_mat ** self.factor["iou"]) * (cls_mat ** self.factor["cls"])

        # choose topk
        topk_targets, topk_mask = self.filter_topk(target_matrix, topk=self.topk)

        # delete one anchor pred assign to mutliple gts
        unique_indices = self.filter_duplicates(topk_targets)

        # TODO: do we need grid_mask? Filter the valid groud truth
        valid_mask = (grid_mask.sum(dim=-2) * topk_mask.sum(dim=-2)).bool()

        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls = torch.gather(target_cls, 1, unique_indices).squeeze(-1)
        align_cls = F.one_hot(align_cls, self.class_num)

        # normalize class ditribution
        max_target = target_matrix.amax(dim=-1, keepdim=True)
        max_iou = iou_mat.amax(dim=-1, keepdim=True)
        normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
        normalize_term = normalize_term.permute(0, 2, 1).gather(2, unique_indices)
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]

        return torch.cat([align_cls, align_bbox], dim=-1), valid_mask.bool()


class Vec2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            logger.info(f"🈶 Found stride of model {anchor_cfg.strides}")
            self.strides = anchor_cfg.strides
        else:
            logger.info("🧸 Found no stride of model, performed a dummy test for auto-anchor size")
            self.strides = self.create_auto_anchor(model, image_size)

        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    def create_auto_anchor(self, model: YOLO, image_size):
        dummy_input = torch.zeros(1, 3, *image_size).to(self.device)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(image_size[1] // anchor_num[1])
        return strides

    def update(self, image_size):
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.anchor_grid, self.scaler = anchor_grid.to(self.device), scaler.to(self.device)

    def __call__(self, predicts):
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in predicts:
            pred_cls, pred_anc, pred_box = layer_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        return preds_cls, preds_anc, preds_box


class Anc2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            logger.info(f"🈶 Found stride of model {anchor_cfg.strides}")
            self.strides = anchor_cfg.strides
        else:
            logger.info("🧸 Found no stride of model, performed a dummy test for auto-anchor size")
            self.strides = self.create_auto_anchor(model, image_size)

        self.head_num = len(anchor_cfg.anchor)
        self.anchor_grids = self.generate_anchors(image_size)
        self.anchor_scale = tensor(anchor_cfg.anchor, device=device).view(self.head_num, 1, -1, 1, 1, 2)
        self.anchor_num = self.anchor_scale.size(2)
        self.class_num = model.num_classes

    def create_auto_anchor(self, model: YOLO, image_size):
        dummy_input = torch.zeros(1, 3, *image_size).to(self.device)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head.shape
            strides.append(image_size[1] // anchor_num[1])
        return strides

    def generate_anchors(self, image_size: List[int]):
        anchor_grids = []
        for stride in self.strides:
            W, H = image_size[0] // stride, image_size[1] // stride
            anchor_h, anchor_w = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
            anchor_grid = torch.stack((anchor_w, anchor_h), 2).view((1, 1, H, W, 2)).float().to(self.device)
            anchor_grids.append(anchor_grid)
        return anchor_grids

    def update(self, image_size):
        self.anchor_grid = self.generate_anchors(image_size)

    def __call__(self, predicts: List[Tensor]):
        preds_box, preds_cls, preds_cnf = [], [], []
        for layer_idx, predict in enumerate(predicts):
            predict = rearrange(predict, "B (L C) h w -> B L h w C", L=self.anchor_num)
            pred_box, pred_cnf, pred_cls = predict.split((4, 1, self.class_num), dim=-1)
            pred_box = pred_box.sigmoid()
            pred_box[..., 0:2] = (pred_box[..., 0:2] * 2.0 - 0.5 + self.anchor_grids[layer_idx]) * self.strides[
                layer_idx
            ]
            pred_box[..., 2:4] = (pred_box[..., 2:4] * 2) ** 2 * self.anchor_scale[layer_idx]
            preds_box.append(rearrange(pred_box, "B L h w A -> B (L h w) A"))
            preds_cls.append(rearrange(pred_cls, "B L h w C -> B (L h w) C"))
            preds_cnf.append(rearrange(pred_cnf, "B L h w C -> B (L h w) C"))

        preds_box = torch.concat(preds_box, dim=1)
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_cnf = torch.concat(preds_cnf, dim=1)

        preds_box = transform_bbox(preds_box, "xycwh -> xyxy")
        return preds_cls, None, preds_box, preds_cnf.sigmoid()


def create_converter(model_version: str = "v9-c", *args, **kwargs) -> Union[Anc2Box, Vec2Box]:
    if "v7" in model_version:  # check model if v7
        converter = Anc2Box(*args, **kwargs)
    else:
        converter = Vec2Box(*args, **kwargs)
    return converter


def bbox_nms(cls_dist: Tensor, bbox: Tensor, nms_cfg: NMSConfig, confidence: Optional[Tensor] = None):
    cls_dist = cls_dist.sigmoid() * (1 if confidence is None else confidence)

    # filter class by confidence
    cls_val, cls_idx = cls_dist.max(dim=-1, keepdim=True)
    valid_mask = cls_val > nms_cfg.min_confidence
    valid_cls = cls_idx[valid_mask].float()
    valid_con = cls_val[valid_mask].float()
    valid_box = bbox[valid_mask.repeat(1, 1, 4)].view(-1, 4)

    batch_idx, *_ = torch.where(valid_mask)
    nms_idx = batched_nms(valid_box, valid_cls, batch_idx, nms_cfg.min_iou)
    predicts_nms = []
    for idx in range(cls_dist.size(0)):
        instance_idx = nms_idx[idx == batch_idx[nms_idx]]

        predict_nms = torch.cat(
            [valid_cls[instance_idx][:, None], valid_box[instance_idx], valid_con[instance_idx][:, None]], dim=-1
        )

        predicts_nms.append(predict_nms)
    return predicts_nms


def calculate_map(predictions, ground_truths, iou_thresholds=arange(0.5, 1, 0.05)) -> Dict[str, Tensor]:
    # TODO: Refactor this block, Flexible for calculate different mAP condition?
    device = predictions.device
    n_preds = predictions.size(0)
    n_gts = (ground_truths[:, 0] != -1).sum()
    ground_truths = ground_truths[:n_gts]
    aps = []

    ious = calculate_iou(predictions[:, 1:-1], ground_truths[:, 1:])  # [n_preds, n_gts]

    for threshold in iou_thresholds:
        tp = torch.zeros(n_preds, device=device, dtype=bool)

        max_iou, max_indices = ious.max(dim=1)
        above_threshold = max_iou >= threshold
        matched_classes = predictions[:, 0] == ground_truths[max_indices, 0]
        max_match = torch.zeros_like(ious)
        max_match[arange(n_preds), max_indices] = max_iou
        if max_match.size(0):
            tp[max_match.argmax(dim=0)] = True
        tp[~above_threshold | ~matched_classes] = False

        _, indices = torch.sort(predictions[:, 1], descending=True)
        tp = tp[indices]

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(~tp, dim=0)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (n_gts + 1e-6)

        precision = torch.cat([torch.ones(1, device=device), precision, torch.zeros(1, device=device)])
        recall = torch.cat([torch.zeros(1, device=device), recall, torch.ones(1, device=device)])

        precision, _ = torch.cummax(precision.flip(0), dim=0)
        precision = precision.flip(0)

        ap = torch.trapezoid(precision, recall)
        aps.append(ap)

    mAP = {
        "mAP.5": aps[0],
        "mAP.5:.95": torch.mean(torch.stack(aps)),
    }
    return mAP
