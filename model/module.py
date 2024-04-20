import torch
import torch.nn as nn

# basic

class Conv(nn.Module):
    # basic convlution
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, act=nn.ReLU(),
                 bias=False, auto_padding=True, padding_mode='zeros'):
        
        super().__init__()

        # not yet handle the case when dilation is a tuple
        if auto_padding:
            if isinstance(kernel_size, int):
                padding = (dilation * (kernel_size - 1) + 1) // 2
            else:
                padding = [(dilation * (k - 1) + 1) // 2 for k in kernel_size]

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    # to be implement
    # def fuse_conv_bn(self):


# RepVGG

class RepConv(nn.Module):
    # https://github.com/DingXiaoH/RepVGG
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, groups=1, act=nn.ReLU()):

        super().__init__()

        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride, groups=groups, act=False)
        self.conv2 = Conv(in_channels, out_channels, 1, stride, groups=groups, act=False)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    # to be implement
    # def fuse_convs(self):


# ResNet

class Res(nn.Module):
    # ResNet bottleneck
    def __init__(self, in_channels, out_channels, 
                 groups=1, act=nn.ReLU(), ratio=0.25):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, h_channels, 1, 1, act=act)
        self.cv2 = Conv(h_channels, h_channels, 3, 1, groups=groups, act=act)
        self.cv3 = Conv(h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x)))


class RepRes(nn.Module):
    # RepResNet bottleneck
    def __init__(self, in_channels, out_channels, 
                 groups=1, act=nn.ReLU(), ratio=0.25):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, h_channels, 1, 1, act=act)
        self.cv2 = RepConv(h_channels, h_channels, 3, 1, groups=groups, act=act)
        self.cv3 = Conv(h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x)))


class ConvBlock(nn.Module):
    # ConvBlock
    def __init__(self, in_channels,
                 repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, in_channels, 3, 1, act=act) if repeat == 1 else Conv(in_channels, h_channels, 3, 1, act=act)
        self.cb = nn.Sequential(*(Conv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat-2))) if repeat > 2 else nn.Identity()
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return self.cv2(self.cb(self.cv1(x)))


class RepConvBlock(nn.Module):
    # ConvBlock
    def __init__(self, in_channels,
                 repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, in_channels, 3, 1, act=act) if repeat == 1 else RepConv(in_channels, h_channels, 3, 1, act=act)
        self.cb = nn.Sequential(*(RepConv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat-2))) if repeat > 2 else nn.Identity()
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return self.cv2(self.cb(self.cv1(x)))


class ResConvBlock(nn.Module):
    # ResConvBlock
    def __init__(self, in_channels, 
                 repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, in_channels, 3, 1, act=act) if repeat == 1 else Conv(in_channels, h_channels, 3, 1, act=act)
        self.cb = nn.Sequential(*(Conv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat-2))) if repeat > 2 else nn.Identity()
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return x + self.cv2(self.cb(self.cv1(x)))


class ResRepConvBlock(nn.Module):
    # ResConvBlock
    def __init__(self, in_channels,
                 repeat=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, in_channels, 3, 1, act=act) if repeat == 1 else RepConv(in_channels, h_channels, 3, 1, act=act)
        self.cb = nn.Sequential(*(RepConv(in_channels, in_channels, 3, 1, act=act) for _ in range(repeat-2))) if repeat > 2 else nn.Identity()
        self.cv2 = nn.Identity() if repeat == 1 else Conv(h_channels, in_channels, 3, 1, act=act)

    def forward(self, x):
        return x + self.cv2(self.cb(self.cv1(x)))


# Darknet

class Dark(nn.Module):
    # DarkNet bottleneck
    def __init__(self, in_channels, out_channels,
                 groups=1, act=nn.ReLU(), ratio=0.5):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = Conv(in_channels, h_channels, 1, 1, act=act)
        self.cv2 = Conv(h_channels, out_channels, 3, 1, groups=groups, act=act)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class RepDark(nn.Module):
    # RepDarkNet bottleneck
    def __init__(self, in_channels, out_channels,
                 groups=1, act=nn.ReLU(), ratio=0.5):

        super().__init__()

        h_channels = int(in_channels * ratio)
        self.cv1 = RepConv(in_channels, h_channels, 3, 1, groups=groups, act=act)
        self.cv2 = Conv(h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


# CSPNet

class CSP(nn.Module):
    # CSPNet
    def __init__(self, in_channels, out_channels,
                 repeat=1, cb_repeat=2, act=nn.ReLU(), ratio=1.0):

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
    def __init__(self, in_channels, out_channels,
                 repeat=1, groups=1, act=nn.ReLU(), ratio=1.0):

        super().__init__()

        h_channels = in_channels // 2
        self.cv1 = Conv(in_channels, in_channels, 1, 1, act=act)
        self.cb = nn.Sequential(*(Dark(h_channels, h_channels, groups=groups, act=act, ratio=ratio) for _ in range(repeat)))
        self.cv2 = Conv(2 * h_channels, out_channels, 1, 1, act=act)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))

        return self.cv2(torch.cat((self.cb(y[0]), y[1]), 1))


# ELAN

class ELAN(nn.Module):
    # ELAN
    def __init__(self, in_channels, out_channels, med_channels,
                 elan_repeat=2, cb_repeat=2, ratio=1.0):

        super().__init__()

        h_channels = med_channels // 2
        self.cv1 = Conv(in_channels, med_channels, 1, 1)
        self.cb = nn.ModuleList(ConvBlock(h_channels, repeat=cb_repeat, ratio=ratio) for _ in range(elan_repeat))
        self.cv2 = Conv((2+elan_repeat) * h_channels, out_channels, 1, 1)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in self.cb)

        return self.cv2(torch.cat(y, 1))


class CSPELAN(nn.Module):
    # ELAN
    def __init__(self, in_channels, out_channels, med_channels,
                 elan_repeat=2, cb_repeat=2, ratio=1.0):

        super().__init__()

        h_channels = med_channels // 2
        self.cv1 = Conv(in_channels, med_channels, 1, 1)
        self.cb = nn.ModuleList(CSP(h_channels, h_channels, repeat=cb_repeat, ratio=ratio) for _ in range(elan_repeat))
        self.cv2 = Conv((2+elan_repeat) * h_channels, out_channels, 1, 1)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in self.cb)

        return self.cv2(torch.cat(y, 1))
