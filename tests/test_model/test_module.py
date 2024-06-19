import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.model.module import SPPELAN, ADown, CBLinear, Conv, Pool

STRIDE = 2
KERNEL_SIZE = 3
IN_CHANNELS = 64
OUT_CHANNELS = 128
NECK_CHANNELS = 64


def test_conv():
    conv = Conv(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = conv(x)
    assert out.shape == (1, OUT_CHANNELS, 64, 64)


def test_pool_max():
    pool = Pool("max", 2, stride=2)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = pool(x)
    assert out.shape == (1, IN_CHANNELS, 32, 32)


def test_pool_avg():
    pool = Pool("avg", 2, stride=2)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = pool(x)
    assert out.shape == (1, IN_CHANNELS, 32, 32)


def test_adown():
    adown = ADown(IN_CHANNELS, OUT_CHANNELS)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = adown(x)
    assert out.shape == (1, OUT_CHANNELS, 32, 32)


def test_cblinear():
    cblinear = CBLinear(IN_CHANNELS, [5, 5])
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    outs = cblinear(x)
    assert len(outs) == 2
    assert outs[0].shape == (1, 5, 64, 64)
    assert outs[1].shape == (1, 5, 64, 64)


def test_sppelan():
    sppelan = SPPELAN(IN_CHANNELS, OUT_CHANNELS, NECK_CHANNELS)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = sppelan(x)
    assert out.shape == (1, OUT_CHANNELS, 64, 64)
