import time
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config
from yolo.tools.bbox_helper import (
    BoxMatcher,
    calculate_iou,
    make_anchor,
    transform_bbox,
)


def get_loss_function(*args, **kwargs):
    raise NotImplementedError


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
        self.class_num = cfg.hyper.data.class_num
        self.image_size = list(cfg.hyper.data.image_size)
        self.strides = cfg.model.anchor.strides
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reverse_reg = torch.arange(self.reg_max, dtype=torch.float16, device=device)
        self.scale_up = torch.tensor(self.image_size * 2, device=device)

        self.anchors, self.scaler = make_anchor(self.image_size, self.strides, device)

        self.cls = BCELoss()
        self.dfl = DFLoss(self.anchors, self.scaler, self.reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(cfg.hyper.train.matcher, self.class_num, self.anchors)

    def parse_predicts(self, predicts: List[Tensor]) -> Tensor:
        """
        args:
            [B x AnchorClass x h1 x w1, B x AnchorClass x h2 x w2, B x AnchorClass x h3 x w3] // AnchorClass = 4 * 16 + 80
        return:
            [B x HW x ClassBbox] // HW = h1*w1 + h2*w2 + h3*w3, ClassBox = 80 + 4 (xyXY)
        """
        preds = []
        for pred in predicts:
            preds.append(rearrange(pred, "B AC h w -> B (h w) AC"))  # B x AC x h x w-> B x hw x AC
        preds = torch.concat(preds, dim=1)  # -> B x (H W) x AC

        preds_anc, preds_cls = torch.split(preds, (self.reg_max * 4, self.class_num), dim=-1)
        preds_anc = rearrange(preds_anc, "B  hw (P R)-> B hw P R", P=4)

        pred_LTRB = preds_anc.softmax(dim=-1) @ self.reverse_reg * self.scaler.view(1, -1, 1)

        lt, rb = pred_LTRB.chunk(2, dim=-1)
        pred_minXY = self.anchors - lt
        pred_maxXY = self.anchors + rb
        predicts = torch.cat([preds_cls, pred_minXY, pred_maxXY], dim=-1)

        return predicts, preds_anc

    def parse_targets(self, targets: Tensor, batch_size: int = 16) -> List[Tensor]:
        """
        return List:
        """
        targets[:, 2:] = transform_bbox(targets[:, 2:], "xycwh -> xyxy") * self.scale_up
        bbox_num = targets[:, 0].int().bincount()
        batch_targets = torch.zeros(batch_size, bbox_num.max(), 5, device=targets.device)
        for instance_idx, bbox_num in enumerate(bbox_num):
            instance_targets = targets[targets[:, 0] == instance_idx]
            batch_targets[instance_idx, :bbox_num] = instance_targets[:, 1:].detach()
        return batch_targets

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.scaler[None, :, None]
        return anchors_cls, anchors_box

    @torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Batch_Size x (Anchor + Class) x H x W
        # TODO: check datatype, why targets has a little bit error with origin version
        predicts, predicts_anc = self.parse_predicts(predicts[0])
        targets = self.parse_targets(targets, batch_size=predicts.size(0))

        align_targets, valid_masks = self.matcher(targets, predicts)
        # calculate loss between with instance and predict

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_cls, predicts_bbox = self.separate_anchor(predicts)

        cls_norm = targets_cls.sum()
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_bbox, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        logger.info("Loss IoU: {:.5f}, DFL: {:.5f}, CLS: {:.5f}", loss_iou, loss_dfl, loss_cls)
        return loss_iou, loss_dfl, loss_cls
