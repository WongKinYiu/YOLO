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
    model = create_model(cfg.model, weight_path=None)
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
        torch.Size([1, 1, 80, 40, 2]),
        torch.Size([1, 1, 40, 20, 2]),
        torch.Size([1, 1, 20, 10, 2]),
    ]
    assert anc2box.anchor_scale.shape == torch.Size([3, 1, 3, 1, 1, 2])


def test_bbox_nms():
    cls_dist = torch.tensor(
        [
            [
                [0.7, 0.1, 0.2],  # High confidence, class 0
                [0.3, 0.6, 0.1],  # High confidence, class 1
                [-3.0, -2.0, -1.0],  # low confidence, class 2
                [0.6, 0.2, 0.2],  # Medium confidence, class 0
            ],
            [
                [0.55, 0.25, 0.2],  # Medium confidence, class 0
                [-4.0, -0.5, -2.0],  # low confidence, class 1
                [0.15, 0.2, 0.65],  # Medium confidence, class 2
                [0.8, 0.1, 0.1],  # High confidence, class 0
            ],
        ],
        dtype=float32,
    )

    bbox = torch.tensor(
        [
            [
                [0, 0, 160, 120],  # Overlaps with box 4
                [160, 120, 320, 240],
                [0, 120, 160, 240],
                [16, 12, 176, 132],
            ],
            [
                [0, 0, 160, 120],  # Overlaps with box 4
                [160, 120, 320, 240],
                [0, 120, 160, 240],
                [16, 12, 176, 132],
            ],
        ],
        dtype=float32,
    )

    nms_cfg = NMSConfig(min_confidence=0.5, min_iou=0.5, max_bbox=400)

    # Batch 1:
    #  - box 1 is kept with classes 0 and 2 as it overlaps with box 4 and has a higher confidence for classes 0 and 2.
    #  - box 2 is kept with classes 0, 1, 2 as it does not overlap with any other box.
    #  - box 3 is rejected by the confidence filter.
    #  - box 4 is kept with class 1 as it overlaps with box 1 and has a higher confidence for class 1.
    # Batch 2:
    #  - box 1 is kept with classes 1 and 2 as it overlaps with box 1 and has a higher confidence for classes 1 and 2.
    #  - box 2 is rejected by the confidence filter.
    #  - box 3 is kept with classes 0, 1, 2 as it does not overlap with any other box.
    #  - box 4 is kept with class 0 as it overlaps with box 1 and has a higher confidence for class 0.
    expected_output = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 160.0, 120.0, 0.6682],
                [1.0, 160.0, 120.0, 320.0, 240.0, 0.6457],
                [0.0, 160.0, 120.0, 320.0, 240.0, 0.5744],
                [2.0, 0.0, 0.0, 160.0, 120.0, 0.5498],
                [1.0, 16.0, 12.0, 176.0, 132.0, 0.5498],
                [2.0, 160.0, 120.0, 320.0, 240.0, 0.5250],
            ],
            [
                [0.0, 16.0, 12.0, 176.0, 132.0, 0.6900],
                [2.0, 0.0, 120.0, 160.0, 240.0, 0.6570],
                [1.0, 0.0, 0.0, 160.0, 120.0, 0.5622],
                [2.0, 0.0, 0.0, 160.0, 120.0, 0.5498],
                [1.0, 0.0, 120.0, 160.0, 240.0, 0.5498],
                [0.0, 0.0, 120.0, 160.0, 240.0, 0.5374],
            ],
        ]
    )

    output = bbox_nms(cls_dist, bbox, nms_cfg)
    for out, exp in zip(output, expected_output):
        assert allclose(out, exp, atol=1e-4), f"Output: {out} Expected: {exp}"


def test_bbox_nms_float16_precision():
    """
    Test that bbox_nms correctly handles float16 inputs with large coordinates.
    
    The bug: batched_nms internally shifts boxes by (label * max_coord) to separate groups.
    With float16, large coordinates (~3500) combined with large idxs (computed by 
    `batch_idx + valid_cls * bbox.size(0)` in box_nms) cause precision loss,
    making overlapping boxes appear non-overlapping → NMS fails to suppress duplicates.
    
    This test ensures such an extreme cases don't break box_nms implementation.
    """

    # Large coordinates simulating real-world high-res image detections (~3500 range)
    # Two clusters of heavily overlapping boxes per image, per class
    # These SHOULD be suppressed to 1 box per cluster per class
    cls_dist = torch.tensor(
        [
            [
                # anchor 0-7: cluster A (x≈3428), high conf class 2
                [-10, -10, 2.0],
                [-10, -10, 0.8],
                [-10, -10, 0.5],
                [-10, -10, 0.2],
                # anchor 4-7: cluster B (x≈2056), high conf class 2
                [-10, -10, 0.8],
                [-10, -10, 1.6],
                [-10, -10, 0.3],
                [-10, -10, 0.2],
            ]
        ] * 8,
        dtype=torch.float16,
    ).to("cuda")

    bbox = torch.tensor(
        [
            [
                # Cluster A: tightly overlapping boxes around x≈3428, y≈85-300
                # IoU between any pair >> 0.5, should suppress to 1
                [3428.0, 85.0625, 3500.0, 298.7500],
                [3428.0, 85.9375, 3500.0, 295.5000],
                [3428.0, 93.0625, 3500.0, 294.0000],
                [3428.0, 92.1875, 3500.0, 293.0000],
                # Cluster B: tightly overlapping boxes around x≈2056, y≈756-918
                # IoU between any pair >> 0.5, should suppress to 1
                # IoU between cluster A and B = 0.0 (non-overlapping) → both kept
                [2056.0, 757.0000, 2392.0, 917.5000],
                [2054.0, 756.5000, 2392.0, 918.0000],
                [2058.0, 756.0000, 2392.0, 916.5000],
                [2054.0, 756.0000, 2392.0, 915.5000],
            ]
        ] * 8,
        dtype=torch.float16,
    ).to("cuda")

    nms_cfg = NMSConfig(min_confidence=0.5, min_iou=0.5, max_bbox=400)

    # Expected: for each image, class 2 has 2 objects (cluster A and cluster B)
    # → exactly 2 boxes per image should survive, both class 2
    # The highest scoring box from each cluster is kept (sigmoid of 2.0 and 1.6)
    #
    # If float16 bug is present: NMS fails to suppress within clusters
    # → more boxes survive per image instead of 2
    output = bbox_nms(cls_dist, bbox, nms_cfg)

    for batch_i, result in enumerate(output):
        num_kept = result.shape[0]
        assert num_kept == 2, (
            f"Image {batch_i}: expected 2 boxes (1 per cluster), got {num_kept}. "
            f"Float16 precision bug likely causing NMS to fail suppression.\n"
            f"Kept boxes:\n{result}"
        )

        kept_classes = result[:, 0]
        assert (kept_classes == 2).all(), (
            f"Image {batch_i}: expected all kept boxes to be class 2, got {kept_classes}"
        )

        kept_boxes = result[:, 1:5]

        # One box should be from cluster A (x1≈3426-3428)
        # One box should be from cluster B (x1≈2054-2058)
        cluster_a_mask = kept_boxes[:, 0] > 3000
        cluster_b_mask = kept_boxes[:, 0] < 3000
        assert cluster_a_mask.sum() == 1, (
            f"Image {batch_i}: expected exactly 1 box from cluster A (x≈3428), "
            f"got {cluster_a_mask.sum()}"
        )
        assert cluster_b_mask.sum() == 1, (
            f"Image {batch_i}: expected exactly 1 box from cluster B (x≈2056), "
            f"got {cluster_b_mask.sum()}"
        )

        # Highest scoring box from each cluster should be kept (sigmoid of 2.0 and 1.6)
        kept_scores = result[:, 5]
        expected_score_a = torch.tensor(2.0, dtype=torch.float16).sigmoid().item()
        expected_score_b = torch.tensor(1.6, dtype=torch.float16).sigmoid().item()
        assert abs(kept_scores[cluster_a_mask].item() - expected_score_a) < 1e-2, (
            f"Image {batch_i}: cluster A score mismatch. "
            f"Got {kept_scores[cluster_a_mask].item():.4f}, expected {expected_score_a:.4f}"
        )
        assert abs(kept_scores[cluster_b_mask].item() - expected_score_b) < 1e-2, (
            f"Image {batch_i}: cluster B score mismatch. "
            f"Got {kept_scores[cluster_b_mask].item():.4f}, expected {expected_score_b:.4f}"
        )


def test_calculate_map():
    predictions = tensor([[0, 60, 60, 160, 160, 0.5], [0, 40, 40, 120, 120, 0.5]])  # [class, x1, y1, x2, y2]
    ground_truths = tensor([[0, 50, 50, 150, 150], [0, 30, 30, 100, 100]])  # [class, x1, y1, x2, y2]

    mAP = calculate_map(predictions, ground_truths)
    expected_ap50 = tensor(0.5050)
    expected_ap50_95 = tensor(0.2020)

    assert isclose(mAP["map_50"], expected_ap50, atol=1e-4), f"AP50 mismatch"
    assert isclose(mAP["map"], expected_ap50_95, atol=1e-4), f"Mean AP mismatch"
