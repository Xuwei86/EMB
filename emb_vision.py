#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import torch.nn.functional as F
import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import  nn , Tensor
# from torch.nn import functional as F

from cvnets.layers import ConvLayer2d, get_normalization_layer
from cvnets.modules.base_module import BaseModule
from cvnets.modules.transformer import LinearAttnFFN, TransformerEncoder
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x

class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

from .modules import NonLocalBlock
from .test_modules import SE, CA, ECA, ECA2, CACA, SASA, ELA, iRMB, LayerNorm
class SCFB(BaseModule):
    """
    """
    def __init__(self,
                 dim,
                 in_channels,
                 stride = 1,
                 drop_path_ratio_0=0.1,
                 drop_path_ratio_1=0.1,
                 mv_expand_ratio = 4,
                 # group = 8,
       ):
        super(MoConvBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn0 = WSA(in_channels//2, in_channels//2)
        self.attn1 = CA(in_channels // 2,  ratio=16)
        self.drop_path_0 = DropPath(drop_path_ratio_0) if drop_path_ratio_0 > 0. else nn.Identity()
        self.drop_path_1 = DropPath(drop_path_ratio_1) if drop_path_ratio_1 > 0. else nn.Identity()

        self.norm2 = LayerNorm(dim)
        self.mlp= InvertedResidual( in_channels=in_channels, out_channels=in_channels, stride=stride, expand_ratio = mv_expand_ratio )

    def forward(self, x):

        x = self.norm1(x)
        #
        x0, x1 = x.chunk(2, dim=1)
        x0 = self.attn0(x0)
        x1 = self.attn1(x1)
        attn = torch.cat([x0, x1], dim=1)
        x = x + self.drop_path_0(attn)

        x = x + self.drop_path_1(self.mlp(self.norm2(x)))

        return x

from mamba_ssm import Mamba

class MFB(BaseModule):
    """
    """
    def __init__(self,
                 dim,
                 window_size,
                 in_channels,
                 stride = 1,
                 drop_path_ratio_0=0.1,
                 drop_path_ratio_1=0.1,
                 mv_expand_ratio = 4,
                 groups = 4,
                 conv_bias=True,
                 device=None,
                 dtype=None,
       ):
        super(MFB, self).__init__()
        self.in_channels = in_channels
        self.norm1 = LayerNorm(dim)
        self.window_size = window_size

        self.attn0 = WSA(in_channels//2, in_channels//2)

        self.attn1 = ECA(in_channels // 2)

        self.conv = ConvLayer(
            in_channels=in_channels//2,
            out_channels=in_channels//2,
            kernel_size=3,
            groups=in_channels//2
        )
        self.mamba = Mamba(
            d_model=dim//2,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=3,  #  4 Local convolution width
            expand=1 # Block expansion factor
        )

        self.drop_path_0 = DropPath(drop_path_ratio_0) if drop_path_ratio_0 > 0. else nn.Identity()
        self.drop_path_1 = DropPath(drop_path_ratio_1) if drop_path_ratio_1 > 0. else nn.Identity()

        self.norm2 = LayerNorm(dim)
        self.mlp = InvertedResidual( in_channels=in_channels, out_channels=in_channels, stride=stride, expand_ratio = mv_expand_ratio )

        self.groups = groups

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        # b, c, l = x.shape
        x = x.reshape(b, groups, c//groups, h, w)
        x = x.permute(0, 2, 1, 3, 4 )

        # flatten
        x = x.reshape(b, c, h, w)
        return x
    def forward(self, x):

        x_i = x
        x_n = self.norm1(x)
        x0, x1 = x.chunk(2, dim=1)
        x0 = self.attn0(x0)
        B, C, H, W = x1.shape
        x10 = x1
        x10 = self.attn1(x10)

        x1 = self.conv(x1)
        n_tokens = x1.shape[2:].numel()
        img_dims = x1.shape[2:]
        x1_flat = x1.reshape(B, C, n_tokens).transpose(-1, -2)

        x1_flip_l = torch.flip(x1_flat, dims=[2])
        x1_flip_c = torch.flip(x1_flat, dims=[1])
        x1_flip_lc = torch.flip(x1_flat, dims=[1,2])

        x1_ori = self.mamba(x1_flat)
        x1_mamba_l = self.mamba(x1_flip_l)
        x1_mamba_c = self.mamba(x1_flip_c)
        x1_mamba_lc = self.mamba(x1_flip_lc)

        x1_ori_l = torch.flip(x1_mamba_l, dims=[2])
        x1_ori_c = torch.flip(x1_mamba_c, dims=[1])
        x1_ori_lc = torch.flip(x1_mamba_lc, dims=[1,2])
        x1_mamba = (x1_ori+x1_ori_l+x1_ori_c+x1_ori_lc)/4

        x1_mamba = x1_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        # out= self.channel_shuffle(x1_mamba, groups=self.groups)
        out = x1_mamba + x10
        out = self.channel_shuffle(out, groups=self.groups)

        attn = torch.cat([x0, out], dim=1)

        x = x + self.drop_path_0(attn)
        x = x_i + self.mlp(self.norm2(x))
        return x


