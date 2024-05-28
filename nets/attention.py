import torch
import math
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import functional as F
from torch.autograd import Variable

torch_ver = torch.__version__[:3]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=size, padding=size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class SELayer(nn.Module):
    '''
    Original SE block, details refer to "Jie Hu et al.: Squeeze-and-Excitation Networks"
    '''
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias = False),
                                nn.ELU(inplace = True),
                                nn.Linear(channel // reduction, channel, bias = False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)###view为reshape功能
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)##expand_as 扩展张量尺寸

class MultiHeadAttentions(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttentions, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        # 线性层用于将输入转换为查询、键和值
        self.linear_q = nn.Linear(in_channels, in_channels)
        self.linear_k = nn.Linear(in_channels, in_channels)
        self.linear_v = nn.Linear(in_channels, in_channels)
        # 最后的线性层用于融合多头注意力的结果
        self.linear_out = nn.Linear(in_channels, in_channels)
        self.layernorm = nn.LayerNorm(in_channels)


    def forward(self, x):

        batch_size, in_channels, height, width = x.size()
        # 将输入张量分别转换为查询、键和值
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)

        query = query.view(batch_size, height, width, in_channels).permute(0, 3, 1, 2)
        key = key.view(batch_size, height, width, in_channels).permute(0, 3, 1, 2)
        value = value.view(batch_size, height, width, in_channels).permute(0, 3, 1, 2)

        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value).permute(0, 2, 3, 1).contiguous().view(batch_size, height, width, in_channels).permute(0, 3, 1, 2)
        attention_output = self.layernorm(attention_output + x)

        output = self.linear_out(attention_output)
        return output


class GlobalFilter(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.size()
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        x = x.reshape(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, C, H, W)

        return x

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, att_name='FHAM', norm_layer=nn.BatchNorm2d):   # norm_layer：用于归一化的层，默认为二维批归一化层
        super(Attention, self).__init__()

        self.att_name = att_name

        if self.att_name == 'CBAM':                   # 通道注意力模块ca和空间注意力模块sa
            self.ca = ChannelAttention(in_channels)
            self.sa = SpatialAttention()
        elif self.att_name == 'DA':                   # 位置注意力模块sa和通道注意力模块sc
            self.sa = PAM_Module(in_channels)
            self.sc = CAM_Module(in_channels)
        else:                                          # 通道注意力模块ca、空间注意力模块sa和SE模块sq
            assert self.att_name == 'FHAM'
            self.glo = GlobalFilter(in_channels, h=64, w=33)
            self.ca = ChannelAttention(in_channels)
            self.sa = SpatialAttention()
            self.sq = SELayer(in_channels)

    def forward(self, x):
        if self.att_name == 'CBAM':
            x = self.ca(x) * x
            x = self.sa(x) * x
            out = x
        elif self.att_name == 'DA':
            sa_feat = self.sa(x)
            sc_feat = self.sc(x)
            out = sa_feat + sc_feat
        else:
            assert self.att_name == 'FHAM'
            x = self.glo(x)
            x1 = self.ca(x) * x
            x2 = self.sa(x1) * x1
            x3 = self.sq(x)
            out = x2 + x3
        return out


