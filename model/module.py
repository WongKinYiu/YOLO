from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def auto_pad(
    kernel: Union[int, Tuple[int, int]], dilation: Union[int, Tuple[int, int]] = 1, **kwargs
) -> Tuple[int, int]:
    """
    Auto Padding for the convolution blocks
    """
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def get_activation(activation: str) -> nn.Module:
    """
    Retrieves an activation function from the PyTorch nn module based on its name, case-insensitively.
    """

    activation_map = {
        name.lower(): obj
        for name, obj in nn.modules.activation.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, nn.Module)
    }

    # Look up the activation function case-insensitively
    if activation.lower() in activation_map:
        return activation_map[activation.lower()]()
    else:
        raise ValueError(f"Activation function '{activation}' is not found in torch.nn")


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class Conv(nn.Module):
    """
    Basic convolutional block with batch normalization and activation.

    Args:
        act (str): The activation function to use. Defaults to "SiLU".
        **kwargs: Additional keyword arguments for the nn.Conv2d layer.
    """

    def __init__(self, act: str = "SiLU", **kwargs) -> None:
        super().__init__()
        out_channels = kwargs.get("out_channels")
        if out_channels is None:
            raise ValueError("out_channels must be specified in kwargs")

        self.conv = nn.Conv2d(**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# RepVGG
class RepConv(nn.Module):
    # https://github.com/DingXiaoH/RepVGG
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding=None, stride=1, groups=1, act=nn.SiLU(), deploy=False
    ):

        super().__init__()
        self.deploy = deploy
        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride, groups=groups, act=False)
        self.conv2 = Conv(in_channels, out_channels, 1, stride, groups=groups, act=False)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    # to be implement
    # def fuse_convs(self):
    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv


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
    def __init__(self, in_channels, out_channels, repeat=1, cb_repeat=2, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = in_channels // 2
        self.cv1 = Conv(in_channels, in_channels, 1, 1, act=act)
        self.cb = nn.Sequential(*(ResConvBlock(h_channels, act=act, repeat=cb_repeat) for _ in range(repeat)))
        self.cv2 = Conv(2 * h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))

        return self.cv2(torch.cat((self.cb(y[0]), y[1]), 1))


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


class MaxPool(nn.Module):
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.pool_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_layer(x)


# TODO: check if Mit
class SPPCSPConv(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPConv, self).__init__()
        c_ = int(2 * out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, out_channels, 1, 1)

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


class UpSample(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.UpSample = nn.Upsample(**kwargs)

    def forward(self, x):
        return self.UpSample(x)


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
