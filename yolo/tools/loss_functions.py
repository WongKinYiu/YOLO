from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config, LossConfig
from yolo.utils.bounding_box_utils import BoxMatcher, Vec2Box, calculate_iou


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

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
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
        )
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
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box.anchor_grid)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = targets_cls.sum()
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls


class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

    def __call__(
        self, aux_predicts: List[Tensor], main_predicts: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        loss_dict = {
            "BoxLoss": self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            "DFLoss": self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            "BCELoss": self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        }
        loss_sum = sum(list(loss_dict.values())) / len(loss_dict)
        return loss_sum, loss_dict


def create_loss_function(cfg: Config, vec2box) -> DualLoss:
    # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
    loss_function = DualLoss(cfg, vec2box)
    logger.info("✅ Success load loss function")
    return loss_function
