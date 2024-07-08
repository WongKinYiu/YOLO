import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from torch import allclose, float32, isclose, tensor

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from yolo import Config, NMSConfig, create_model
from yolo.config.config import AnchorConfig
from yolo.utils.bounding_box_utils import (
    Anc2Box,
    Vec2Box,
    bbox_nms,
    calculate_iou,
    calculate_map,
    generate_anchors,
    transform_bbox,
)

EPS = 1e-4


@pytest.fixture
def dummy_bboxes():
    bbox1 = tensor([[50, 80, 150, 140], [30, 20, 100, 80]], dtype=float32)
    bbox2 = tensor([[90, 70, 160, 160], [40, 40, 90, 120]], dtype=float32)
    return bbox1, bbox2


def test_calculate_iou_2d(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1, bbox2)
    expected_iou = tensor([[0.4138, 0.1905], [0.0096, 0.3226]])
    assert iou.shape == (2, 2)
    assert allclose(iou, expected_iou, atol=EPS)


def test_calculate_iou_3d(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1[None], bbox2[None])
    expected_iou = tensor([[0.4138, 0.1905], [0.0096, 0.3226]])
    assert iou.shape == (1, 2, 2)
    assert allclose(iou, expected_iou, atol=EPS)


def test_calculate_diou(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1, bbox2, "diou")
    expected_diou = tensor([[0.3816, 0.0943], [-0.2048, 0.2622]])

    assert iou.shape == (2, 2)
    assert allclose(iou, expected_diou, atol=EPS)


def test_calculate_ciou(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1, bbox2, metrics="ciou")
    # TODO: check result!
    expected_ciou = tensor([[0.3769, 0.0853], [-0.2050, 0.2602]])
    assert iou.shape == (2, 2)
    assert allclose(iou, expected_ciou, atol=EPS)

    bbox1 = tensor([[50, 80, 150, 140], [30, 20, 100, 80]], dtype=float32)
    bbox2 = tensor([[90, 70, 160, 160], [40, 40, 90, 120]], dtype=float32)


def test_transform_bbox_xywh_to_Any(dummy_bboxes):
    bbox1, _ = dummy_bboxes
    transformed_bbox = transform_bbox(bbox1, "xywh -> xyxy")
    expected_bbox = tensor([[50.0, 80.0, 200.0, 220.0], [30.0, 20.0, 130.0, 100.0]])
    assert allclose(transformed_bbox, expected_bbox)


def test_transform_bbox_xycwh_to_Any(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    transformed_bbox = transform_bbox(bbox1, "xycwh -> xycwh")
    assert allclose(transformed_bbox, bbox1)

    transformed_bbox = transform_bbox(bbox2, "xyxy -> xywh")
    expected_bbox = tensor([[90.0, 70.0, 70.0, 90.0], [40.0, 40.0, 50.0, 80.0]])
    assert allclose(transformed_bbox, expected_bbox)


def test_transform_bbox_xyxy_to_Any(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    transformed_bbox = transform_bbox(bbox1, "xyxy -> xyxy")
    assert allclose(transformed_bbox, bbox1)

    transformed_bbox = transform_bbox(bbox2, "xyxy -> xycwh")
    expected_bbox = tensor([[125.0, 115.0, 70.0, 90.0], [65.0, 80.0, 50.0, 80.0]])
    assert allclose(transformed_bbox, expected_bbox)


def test_transform_bbox_invalid_format(dummy_bboxes):
    bbox, _ = dummy_bboxes

    # Test invalid input format
    with pytest.raises(ValueError, match="Invalid input or output format"):
        transform_bbox(bbox, "invalid->xyxy")

    # Test invalid output format
    with pytest.raises(ValueError, match="Invalid input or output format"):
        transform_bbox(bbox, "xywh->invalid")


def test_generate_anchors():
    image_size = [256, 256]
    strides = [8, 16, 32]
    anchors, scalers = generate_anchors(image_size, strides)
    assert anchors.shape[0] == scalers.shape[0]
    assert anchors.shape[1] == 2


def test_vec2box_autoanchor():
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg: Config = compose(config_name="config", overrides=["model=v9-m"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model, weight_path=None).to(device)
    vec2box = Vec2Box(model, cfg.model.anchor, cfg.image_size, device)
    assert vec2box.strides == [8, 16, 32]

    vec2box.update((320, 640))
    assert vec2box.anchor_grid.shape == (4200, 2)
    assert vec2box.scaler.shape == tuple([4200])


def test_anc2box_autoanchor(inference_v7_cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(inference_v7_cfg.model, weight_path=None).to(device)
    anchor_cfg: AnchorConfig = inference_v7_cfg.model.anchor.copy()
    del anchor_cfg.strides
    anc2box = Anc2Box(model, anchor_cfg, inference_v7_cfg.image_size, device)
    assert anc2box.strides == [8, 16, 32]

    anc2box.update((320, 640))
    anchor_grids_shape = [anchor_grid.shape for anchor_grid in anc2box.anchor_grids]
    assert anchor_grids_shape == [
        torch.Size([1, 1, 80, 80, 2]),
        torch.Size([1, 1, 40, 40, 2]),
        torch.Size([1, 1, 20, 20, 2]),
    ]
    assert anc2box.anchor_scale.shape == torch.Size([3, 1, 3, 1, 1, 2])


def test_bbox_nms():
    cls_dist = tensor(
        [[[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]], [[0.4, 0.4, 0.2], [0.5, 0.4, 0.1]]]  # Example class distribution
    )
    bbox = tensor(
        [[[50, 50, 100, 100], [60, 60, 110, 110]], [[40, 40, 90, 90], [70, 70, 120, 120]]],  # Example bounding boxes
        dtype=float32,
    )
    nms_cfg = NMSConfig(min_confidence=0.5, min_iou=0.5)

    expected_output = [
        tensor(
            [
                [1.0000, 50.0000, 50.0000, 100.0000, 100.0000, 0.6682],
                [0.0000, 60.0000, 60.0000, 110.0000, 110.0000, 0.6457],
            ]
        )
    ]

    output = bbox_nms(cls_dist, bbox, nms_cfg)

    for out, exp in zip(output, expected_output):
        assert allclose(out, exp, atol=1e-4), f"Output: {out} Expected: {exp}"


def test_calculate_map():
    predictions = tensor([[0, 60, 60, 160, 160, 0.5], [0, 40, 40, 120, 120, 0.5]])  # [class, x1, y1, x2, y2]
    ground_truths = tensor([[0, 50, 50, 150, 150], [0, 30, 30, 100, 100]])  # [class, x1, y1, x2, y2]

    mAP = calculate_map(predictions, ground_truths)

    expected_ap50 = tensor(0.5)
    expected_ap50_95 = tensor(0.2)

    assert isclose(mAP["mAP.5"], expected_ap50, atol=1e-5), f"AP50 mismatch"
    assert isclose(mAP["mAP.5:.95"], expected_ap50_95, atol=1e-5), f"Mean AP mismatch"
