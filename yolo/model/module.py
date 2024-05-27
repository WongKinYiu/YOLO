from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from yolo.tools.module_helper import auto_pad, get_activation, round_up


# ----------- Basic Class ----------- #
class Conv(nn.Module):
    """A basic convolutional block that includes convolution, batch normalization, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs
    ):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Pool(nn.Module):
    """A generic pooling block supporting 'max' and 'avg' pooling methods."""

    def __init__(self, method: str = "max", kernel_size: _size_2_t = 2, **kwargs):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        pool_classes = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}
        self.pool = pool_classes[method.lower()](kernel_size=kernel_size, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


# ----------- Detection Class ----------- #
class Detection(nn.Module):
    """A single YOLO Detection head for detection models"""

    def __init__(self, in_channels: int, num_classes: int, *, reg_max: int = 16, use_group: bool = True):
        super().__init__()

        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max
        # TODO: round up head[0] channels or each head?
        anchor_neck = max(round_up(in_channels // 4, groups), anchor_channels, 16)
        class_neck = max(in_channels, min(num_classes * 2, 128))

        self.anchor_conv = nn.Sequential(
            Conv(in_channels, anchor_neck, 3),
            Conv(anchor_neck, anchor_neck, 3, groups=groups),
            nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
        )
        self.class_conv = nn.Sequential(
            Conv(in_channels, class_neck, 3), Conv(class_neck, class_neck, 3), nn.Conv2d(class_neck, num_classes, 1)
        )

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        anchor_x = self.anchor_conv(x)
        class_x = self.class_conv(x)
        return torch.cat([anchor_x, class_x], dim=1)


class MultiheadDetection(nn.Module):
    """Mutlihead Detection module for Dual detect or Triple detect"""

    def __init__(self, in_channels: List[int], num_classes: int, **head_kwargs):
        super().__init__()
        self.heads = nn.ModuleList(
            [Detection(head_in_channels, num_classes, **head_kwargs) for head_in_channels in in_channels]
        )

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(x) for x, head in zip(x_list, self.heads)]


# ----------- Backbone Class ----------- #
class RepConv(nn.Module):
    """A convolutional block that combines two convolution layers (kernel and point-wise)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs
    ):
        super().__init__()
        self.act = get_activation(activation)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, activation=False, **kwargs)
        self.conv2 = Conv(in_channels, out_channels, 1, activation=False, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv1(x) + self.conv2(x))


class RepNBottleneck(nn.Module):
    """A bottleneck block with optional residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Tuple[int, int] = (3, 3),
        residual: bool = True,
        expand: float = 0.5,
        **kwargs
    ):
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0], **kwargs)
        self.conv2 = Conv(neck_channels, out_channels, kernel_size[1], **kwargs)
        self.residual = residual

        if residual and (in_channels != out_channels):
            self.residual = False
            logger.warning(
                "Residual connection disabled: in_channels ({}) != out_channels ({})", in_channels, out_channels
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class RepNCSP(nn.Module):
    """RepNCSP block with convolutions, split, and bottleneck processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()

        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size, **kwargs)

        self.bottleneck = nn.Sequential(
            *[RepNBottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.bottleneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class RepNCSPELAN(nn.Module):
    """RepNCSPELAN block combining RepNCSP blocks with ELAN structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        csp_args: Dict[str, Any] = {},
        csp_neck_args: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(torch.cat([x1, x2, x3, x4], dim=1))
        return x5


class ADown(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv1 = Conv(half_in_channels, half_out_channels, **mid_layer)
        self.max_pool = Pool("max", **mid_layer)
        self.conv2 = Conv(half_in_channels, half_out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x2)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)


class CBLinear(nn.Module):
    """Convolutional block that outputs multiple feature maps split along the channel dimension."""

    def __init__(self, in_channels: int, out_channels: List[int], kernel_size: int = 1, **kwargs):
        super(CBLinear, self).__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, sum(out_channels), kernel_size, **kwargs)
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.conv(x)
        return x.split(self.out_channels, dim=1)


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, neck_channels: Optional[int] = None):
        super(SPPELAN, self).__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv(in_channels, neck_channels, kernel_size=1)
        self.pools = nn.ModuleList([Pool("max", 5, stride=1) for _ in range(3)])
        self.conv5 = Conv(4 * neck_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class UpSample(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.UpSample = nn.Upsample(**kwargs)

    def forward(self, x):
        return self.UpSample(x)


############# Waiting For Refactor #############
# ResNet
class Res(nn.Module):
    # ResNet bottleneck
    def __init__(self, in_channels, out_channels, groups=1, act=nn.ReLU(), ratio=0.25):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, h_channels, 1, 1, act=act)
        self.cv2 = Conv(h_channels, h_channels, 3, 1, groups=groups, act=act)
        self.cv3 = Conv(h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x)))


class RepRes(nn.Module):
    # RepResNet bottleneck
    def __init__(self, in_channels, out_channels, groups=1, act=nn.ReLU(), ratio=0.25):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, h_channels, 1, 1, act=act)
        self.cv2 = RepConv(h_channels, h_channels, 3, 1, groups=groups, act=act)
        self.cv3 = Conv(h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x)))


class ConvBlock(nn.Module):
    # ConvBlock
    def __init__(self, in_channels, repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = (
            Conv(in_channels, in_channels, 3, 1, act=act)
            if repeat == 1
            else Conv(in_channels, h_channels, 3, 1, act=act)
        )
        self.cb = (
            nn.Sequential(*(Conv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat - 2)))
            if repeat > 2
            else nn.Identity()
        )
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return self.cv2(self.cb(self.cv1(x)))


class RepConvBlock(nn.Module):
    # ConvBlock
    def __init__(self, in_channels, repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = (
            Conv(in_channels, in_channels, 3, 1, act=act)
            if repeat == 1
            else RepConv(in_channels, h_channels, 3, 1, act=act)
        )
        self.cb = (
            nn.Sequential(*(RepConv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat - 2)))
            if repeat > 2
            else nn.Identity()
        )
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return self.cv2(self.cb(self.cv1(x)))


class ResConvBlock(nn.Module):
    # ResConvBlock
    def __init__(self, in_channels, repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = (
            Conv(in_channels, in_channels, 3, 1, act=act)
            if repeat == 1
            else Conv(in_channels, h_channels, 3, 1, act=act)
        )
        self.cb = (
            nn.Sequential(*(Conv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat - 2)))
            if repeat > 2
            else nn.Identity()
        )
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return x + self.cv2(self.cb(self.cv1(x)))


class ResRepConvBlock(nn.Module):
    # ResConvBlock
    def __init__(self, in_channels, repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = (
            Conv(in_channels, in_channels, 3, 1, act=act)
            if repeat == 1
            else RepConv(in_channels, h_channels, 3, 1, act=act)
        )
        self.cb = (
            nn.Sequential(*(RepConv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat - 2)))
            if repeat > 2
            else nn.Identity()
        )
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return x + self.cv2(self.cb(self.cv1(x)))


# Darknet
class Dark(nn.Module):
    # DarkNet bottleneck
    def __init__(self, in_channels, out_channels, groups=1, act=nn.ReLU(), ratio=0.5):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, h_channels, 1, 1, act=act)
        self.cv2 = Conv(h_channels, out_channels, 3, 1, groups=groups, act=act)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class RepDark(nn.Module):
    # RepDarkNet bottleneck
    def __init__(self, in_channels, out_channels, groups=1, act=nn.ReLU(), ratio=0.5):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = RepConv(in_channels, h_channels, 3, 1, groups=groups, act=act)
        self.cv2 = Conv(h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


# CSPNet
class CSP(nn.Module):
    # CSPNet
    def __init__(self, in_channels, out_channels, repeat=1, cb_repeat=2, act=nn.ReLU()):
        super().__init__()
        h_channels = in_channels // 2
        self.cv1 = Conv(in_channels, in_channels, 1, 1, act=act)
        self.cb = nn.Sequential(*(ResConvBlock(h_channels, act=act, repeat=cb_repeat) for _ in range(repeat)))
        self.cv2 = Conv(2 * h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        x = list(self.cv1(x).chunk(2, 1))
        x = torch.cat((self.cb(x[0]), x[1]), 1)
        x = self.cv2(x)
        return x


class CSPDark(nn.Module):
    # CSPNet
    def __init__(self, in_channels, out_channels, repeat=1, groups=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = in_channels // 2
        self.cv1 = Conv(in_channels, in_channels, 1, 1, act=act)
        self.cb = nn.Sequential(
            *(Dark(h_channels, h_channels, groups=groups, act=act, ratio=ratio) for _ in range(repeat))
        )
        self.cv2 = Conv(2 * h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))

        return self.cv2(torch.cat((self.cb(y[0]), y[1]), 1))


# ELAN
class ELAN(nn.Module):
    # ELAN
    def __init__(self, in_channels, out_channels, med_channels, elan_repeat=2, cb_repeat=2, ratio=1.0):

        super().__init__()

        h_channels = med_channels // 2
        self.cv1 = Conv(in_channels, med_channels, 1, 1)
        self.cb = nn.ModuleList(ConvBlock(h_channels, repeat=cb_repeat, ratio=ratio) for _ in range(elan_repeat))
        self.cv2 = Conv((2 + elan_repeat) * h_channels, out_channels, 1, 1)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in self.cb)

        return self.cv2(torch.cat(y, 1))


class CSPELAN(nn.Module):
    # ELAN
    def __init__(self, in_channels, out_channels, med_channels, elan_repeat=2, cb_repeat=2, ratio=1.0):

        super().__init__()

        h_channels = med_channels // 2
        self.cv1 = Conv(in_channels, med_channels, 1, 1)
        self.cb = nn.ModuleList(CSP(h_channels, h_channels, repeat=cb_repeat, ratio=ratio) for _ in range(elan_repeat))
        self.cv2 = Conv((2 + elan_repeat) * h_channels, out_channels, 1, 1)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in self.cb)

        return self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


# TODO: check if Mit
class SPPCSPConv(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPConv, self).__init__()
        c_ = int(2 * out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1)
        self.cv2 = Conv(in_channels, c_, 1)
        self.cv3 = Conv(c_, c_, 3)
        self.cv4 = Conv(c_, c_, 1)
        self.m = nn.ModuleList([Pool(method="max", kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1)
        self.cv6 = Conv(c_, c_, 3)
        self.cv7 = Conv(2 * c_, out_channels, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class ImplicitA(nn.Module):
    """
    Implement YOLOR - implicit knowledge(Add), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 0.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit + x


class ImplicitM(nn.Module):
    """
    Implement YOLOR - implicit knowledge(multiply), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 1.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit * x


class IDetect(nn.Module):
    """
    #TODO: Add Detect class, change IDetect base class
    """

    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
