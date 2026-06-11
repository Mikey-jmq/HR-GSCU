import torch
import torch.nn.functional as F
from torch import nn

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),  
            nn.BatchNorm2d(inchannel),  
            nn.ReLU(inplace=True)  
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),  
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),  
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False), 
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),  
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),  
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),  
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.convmix(x_f)

        return out


class DEAM(nn.Module):
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(DEAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in  
        self.activation = activation
        self.ds = ds  
        self.pool = nn.AvgPool2d(self.ds)  
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1) 
        self.gamma = nn.Parameter(torch.zeros(1))  
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, input, diff):
        """
            inputs :
                x : 输入特征图 (B X C X W X H)
            returns :
                out : 自注意力值与输入特征相加
                attention: 注意力矩阵 B X N X N (N 是宽度*高度)
        """
        diff = self.pool(diff)  
        m_batchsize, C, width, height = diff.size()
        proj_query = self.query_conv(diff).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(diff).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key) 
        energy = (self.key_channel ** -.5) * energy 
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        x = self.pool(input)  
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input  

        return out

import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:  
            bias = False
        padding = kernel_size // 2 
        layers = list() 
        if transpose: 
            padding = kernel_size // 2 - 1 
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)) 
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))  
        if norm: 
            layers.append(nn.BatchNorm2d(out_channel))  
        if relu:  
            layers.append(nn.GELU()) 
        self.main = nn.Sequential(*layers)  

    def forward(self, x):
        return self.main(x) 

class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()  
        in_channel = nc
        out_channel = nc
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)  
        self.trans_layer = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  
    def forward(self, x):
        out = self.conv1(x) 
        out = self.trans_layer(out)  
        out = self.conv2(out)  
        return out + x 

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction, channel, bias=False),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        b, c, _, _ = x.size() 
        y = self.avg_pool(x).view(b, c)  
        y = self.fc(y).view(b, c, 1, 1) 
        return x * y.expand_as(x) 


class Ddnf(nn.Module):
    def __init__(self, nc):
        super(Ddnf, self).__init__()  
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0), 
            nn.LeakyReLU(0.1, inplace=True), 
            SELayer(channel=nc),  
            nn.Conv2d(nc, nc, 1, 1, 0)  
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  
            nn.LeakyReLU(0.1, inplace=True),  
            SELayer(channel=nc), 
            nn.Conv2d(nc, nc, 1, 1, 0)  
        )

    def forward(self, x):
        _, _, H, W = x.shape 

        x_freq = torch.fft.rfft2(x, norm='backward')  

        ori_mag = torch.abs(x_freq) 
        mag = self.processmag(ori_mag)  
        mag = ori_mag + mag  

        ori_pha = torch.angle(x_freq)  
        pha = self.processpha(ori_pha)  
        pha = ori_pha + pha  

        real = mag * torch.cos(pha)  
        imag = mag * torch.sin(pha)  

        x_out = torch.complex(real, imag) 

        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')  
        return x_freq_spatial  

class BidomainNonlinearMapping(nn.Module):

    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
        self.spatial_process = SpaBlock(in_nc)  
        self.frequency_process = Ddnf(in_nc)  
        self.cat = nn.Conv2d(2 * in_nc, in_nc//2, 1, 1, 0)  

    def forward(self, x):
        _, _, H, W = x.shape  
        x_freq = self.frequency_process(x) 
        x = self.spatial_process(x)  

        xcat = torch.cat([x, x_freq], 1)  
        x_out = self.cat(xcat) 
        return x_out 


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z * res + x
