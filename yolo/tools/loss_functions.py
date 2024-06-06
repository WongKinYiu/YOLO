from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config
from yolo.utils.bounding_box_utils import BoxMatcher, calculate_iou, generate_anchors
from yolo.utils.module_utils import divide_into_chunks


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bce = BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device), reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    def __init__(self, anchors: Tensor, scaler: Tensor, reg_max: int) -> None:
        super().__init__()
        self.anchors = anchors
        self.scaler = scaler
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        anchors_norm = (self.anchors / self.scaler[:, None])[None]
        targets_dist = torch.cat(((anchors_norm - bbox_lt), (bbox_rb - anchors_norm)), -1).clamp(0, self.reg_max - 1.01)
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class YOLOLoss:
    def __init__(self, cfg: Config) -> None:
        self.reg_max = cfg.model.anchor.reg_max
        self.class_num = cfg.model.class_num
        self.image_size = list(cfg.image_size)
        self.strides = cfg.model.anchor.strides
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.anchors, self.scaler = generate_anchors(self.image_size, self.strides)
        self.anchors = self.anchors.to(device)
        self.scaler = self.scaler.to(device)

        self.cls = BCELoss()
        self.dfl = DFLoss(self.anchors, self.scaler, self.reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(cfg.task.loss.matcher, self.class_num, self.anchors)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(
        self, predicts_box: List[Tensor], predicts_anc: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Batch_Size x (Anchor + Class) x H x W
        # TODO: check datatype, why targets has a little bit error with origin version

        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, predicts_box)

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_cls, predicts_bbox = self.separate_anchor(predicts_box)

        cls_norm = targets_cls.sum()
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_bbox, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls


class DualLoss:
    def __init__(self, cfg: Config) -> None:
        self.loss = YOLOLoss(cfg)
        self.aux_rate = cfg.task.loss.aux

        self.iou_rate = cfg.task.loss.objective["BoxLoss"]
        self.dfl_rate = cfg.task.loss.objective["DFLoss"]
        self.cls_rate = cfg.task.loss.objective["BCELoss"]

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:

        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls = self.loss(*predicts[0], targets)
        main_iou, main_dfl, main_cls = self.loss(*predicts[1], targets)

        loss_dict = {
            "BoxLoss": self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            "DFLoss": self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            "BCELoss": self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        }
        loss_sum = sum(list(loss_dict.values())) / len(loss_dict)
        return loss_sum, loss_dict


def get_loss_function(cfg: Config) -> YOLOLoss:
    loss_function = DualLoss(cfg)
    logger.info("✅ Success load loss function")
    return loss_function
