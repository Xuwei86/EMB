import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        # 定义 g、theta、phi、out 四个卷积层
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.out = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)

        # 定义 softmax 层，用于将 f_ij 进行归一化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.size(0)

        # 计算 g(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 计算 theta(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # 计算 phi(x)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 计算 f_ij
        f = torch.matmul(theta_x, phi_x)

        # 对 f_ij 进行归一化
        f_div_C = self.softmax(f)

        # 计算 y_i
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        # 计算 z_i
        y = self.out(y)
        z = y + x

        return z

class SE(nn.Module):
    def __init__(self,channel, ratio=1):
        super(SE,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * ratio, bias=False),
            h_swish(),
            nn.Linear(channel * ratio, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y  = self.fc(y).view(b, c, 1, 1)
        return  x * y


class MLP(nn.Module):
    def __init__(self,in_featrues, ratio=4, act=nn.GELU):
        super().__init__()
        hidden_featrues = in_featrues * ratio
        self.fc1 = nn.Conv2d(in_featrues,hidden_featrues, 1, bias=False)
        self.act = act()
        self.fc2 = nn.Conv2d(hidden_featrues, in_featrues, 1, bias=False)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class CA_atten(nn.Module):
    def __init__(self, inp, ks=7, ratio=2):
        super(CA_atten, self).__init__()

        # mip = max(8, inp // ratio)
        p = ks // 2


        self.conv0 = nn.Conv1d(inp, inp, kernel_size=ks, padding=p, groups=inp, bias=False)
        # self.conv1 = nn.Conv1d(inp, inp, kernel_size=ks, padding=p, groups=1, bias=False)
        self.bn0 = nn.BatchNorm1d(inp)
        # self.bn1 = nn.BatchNorm1d(mip)
        self.sig = nn.Sigmoid()

        self.relu = h_swish()

    def forward(self, x):

        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        identity = x

        # x_h = self.sig(self.bn0(self.conv0(x_h))).view(b, c, h, 1)
        # x_w = self.sig(self.bn0(self.conv0(x_w))).view(b, c, 1, w)

        x_h = self.relu(self.bn0(self.conv0(x_h))).view(b, c, h, 1)
        x_w = self.relu(self.bn0(self.conv0(x_w))).view(b, c, 1, w)

        x_h = self.sig(self.conv0(x_h.view(b, c, h))).view(b, c, h, 1)
        x_w = self.sig(self.conv0(x_w.view(b, c, w))).view(b, c, 1, w)

        y = identity * x_w * x_h

        return y

class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        ks = int(abs((math.log(channel,2) + b) / gamma)) +2
        ks = ks if ks % 2 else ks + 1
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=ks, padding=ks //2, bias = False)
        self.sg = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sg(y).view([b, c, 1, 1])
        return x * y

class DPA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(DPA, self).__init__()
        ks = int(abs((math.log(channel,2) + b) / gamma)) +2
        ks = ks if ks % 2 else ks + 1
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.amg = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=ks, padding=ks //2, bias = False)
        self.sg = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        # a = 1 ; b = 0.2
        y = (self.avg(x)*a + self.amg(x)*b).view([b, 1, c])
        y = self.conv(y)
        y = self.sg(y).view([b, c, 1, 1])
        return x * y

class SASA(nn.Module):
    def __init__(self,  channel, ks=7):
        super(SASA, self).__init__()
        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=channel//8, bias=False)
        # self.gn = nn.GroupNorm(min(channel // 2, 16), channel)
        self.bn = nn.BatchNorm1d(channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        # x_h = self.sig(self.gn(self.conv(x_h))).view(b, c, h, 1)
        # x_w = self.sig(self.gn(self.conv(x_w))).view(b, c, 1, w)
        x_h = self.sig(self.bn(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.sig(self.bn(self.conv(x_w))).view(b, c, 1, w)
        out = x_h * x_w

        return out
class CACA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(CACA, self).__init__()
        ks = int(abs((math.log(channel,2) + b) / gamma)) +2
        ks = ks if ks % 2 else ks + 1
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=ks, padding=ks //2, bias = False)
        self.sg = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sg(y).view([b, c, 1, 1])
        return y

class CA(nn.Module):
    def __init__(self, inp, ratio=16):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // ratio)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, groups=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, groups=1, padding=0)
        self.conv3 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, groups=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class ELA(nn.Module):
    def __init__(self,  channel, ks=7):
        super(ELA, self).__init__()
        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=channel//8, bias=False)
        # self.gn = nn.GroupNorm(min(channel // 2, 16), channel)
        self.bn = nn.BatchNorm1d(channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        # x_h = self.sig(self.gn(self.conv(x_h))).view(b, c, h, 1)
        # x_w = self.sig(self.gn(self.conv(x_w))).view(b, c, 1, w)
        x_h = self.sig(self.bn(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.sig(self.bn(self.conv(x_w))).view(b, c, 1, w)
        out = x * x_h * x_w

        return out

class h_swish(nn.Module):
    def __init__(self,inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)
class h_sigmoid(nn.Module):
    def __init__(self,inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3)/6




import torch.nn.functional as F
from einops import rearrange, reduce
def get_norm(norm_layer='in_1d'):
	eps = 1e-6
	norm_dict = {
		'none': nn.Identity,
		# 'in_1d': partial(nn.InstanceNorm1d, eps=eps),
		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
		# 'in_3d': partial(nn.InstanceNorm3d, eps=eps),
		# 'bn_1d': partial(nn.BatchNorm1d, eps=eps),
		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
		# 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
		# 'bn_3d': partial(nn.BatchNorm3d, eps=eps),
		'gn': partial(nn.GroupNorm, eps=eps),
		'ln_1d': partial(nn.LayerNorm, eps=eps),
		'ln_2d': partial(LayerNorm2d, eps=eps),
        'ln_2d': partial(LayerNorm, eps=eps),
	}
	return norm_dict[norm_layer]
# #
from functools import partial
class ConvNormAct(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

from .test_modules import h_sigmoid
def get_act(act_layer='relu'):
	act_dict = {
		'none': nn.Identity,
		'sigmoid': h_sigmoid,
		'relu': nn.ReLU,
		'silu': nn.SiLU
	}
	return act_dict[act_layer]

class WSA(nn.Module):

    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=8, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        # mamba-t dim_head=8   others 16
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        inplace = True
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=True)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                     act_layer=act_layer, inplace=True)
            else:
                self.v = nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        # self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        # self.proj_drop = nn.Dropout(drop)
        # self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
        # self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        # shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        return x

def get_norm(norm_layer='in_1d'):
	eps = 1e-6
	norm_dict = {
		'none': nn.Identity,
		# 'in_1d': partial(nn.InstanceNorm1d, eps=eps),
		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
		# 'in_3d': partial(nn.InstanceNorm3d, eps=eps),
		# 'bn_1d': partial(nn.BatchNorm1d, eps=eps),
		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
		# 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
		# 'bn_3d': partial(nn.BatchNorm3d, eps=eps),
		'gn': partial(nn.GroupNorm, eps=eps),
		'ln_1d': partial(nn.LayerNorm, eps=eps),
		# 'ln_2d': partial(LayerNorm2d, eps=eps),
        'ln_2d': partial(LayerNorm, eps=eps),
	}
	return norm_dict[norm_layer]

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

from mamba_ssm import Mamba
class TransSSM(nn.Module):
    def __init__(self, groups = 4):
        self.attn1 = ECA(in_channels // 2)
        self.mamba = Mamba(
            d_model=dim // 2,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=3,  # 4 Local convolution width
            expand=1  # Block expansion factor
        )
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        # b, c, l = x.shape
        x = x.reshape(b, groups, c//groups, h, w)
        x = x.permute(0, 2, 1, 3, 4 )

        # flatten
        x = x.reshape(b, c, h, w)
        return x
    def forward(self, x):

        # B, C, H, W = x_1.shape
        B, C, H, W = x.shape
        x0 = x
        x0 = self.attn1(x0)

        x = self.conv(x)
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1,2])

        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)

        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1,2])
        x_mamba = (x_ori+x_ori_l+x_ori_c+x_ori_lc)/4

        x_mamba = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        # out= self.channel_shuffle(x1_mamba, groups=self.groups)
        out = x_mamba + x0
        out = self.channel_shuffle(out, groups=self.groups)

        return out