# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN1 import *
from model.PGCU import *
from model.gscu_final import *
from model.dysample1 import *
from model.SFAU import *

def upsample(x,h,w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, padding=1, bias=False)
        self.relu  = nn.ReLU(True)
        
    def forward(self, x):
        x = x+self.conv2(self.relu(self.conv1(x)))
        return x
    
class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size//2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
            )

    def forward(self, input):
        return self.basic_unit(input)

class LRBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feat):
        super(LRBlock, self).__init__()
        self.get_LR = BasicUnit(ms_channels, n_feat, ms_channels)
        self.get_HR_residual = BasicUnit(ms_channels, n_feat, ms_channels)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels)
        
    def forward(self, HR, LR):
        _,_,M,N = HR.shape
        _,_,m,n = LR.shape
        LR_hat = upsample(self.get_LR(HR), m, n)
        LR_Residual = LR - LR_hat
        HR_Residual = upsample(self.get_HR_residual(LR_Residual), M, N)
        HR = self.prox(HR + HR_Residual)
        return HR
        
class PANBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat, 
                 kernel_size):
        super(PANBlock, self).__init__()
        self.get_PAN = BasicUnit(ms_channels, n_feat, pan_channels, kernel_size)
        self.get_HR_residual = BasicUnit(pan_channels, n_feat, ms_channels, kernel_size)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels, kernel_size)
        
    def forward(self, HR, PAN):
        PAN_hat = self.get_PAN(HR)
        PAN_Residual = PAN - PAN_hat
        HR_Residual = self.get_HR_residual(PAN_Residual)
        HR = self.prox(HR + HR_Residual)
        return HR
        
class GPPNN(nn.Module):
    def __init__(self, 
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN, self).__init__()
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])
        # self.PGCU = PGCU(4, 128)
        # self.gnnmodel = adjGNN(ms_channels)
        # self.Dysample= DySample1(1)
        self.ConvTrans1 = nn.Sequential(nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        # self.recon = nn.Sequential(
        #     nn.Conv2d(ms_channels, n_feat, 3, padding=1, bias=False),
        #     nn.ReLU(True),
        #     *[ResBlock(n_feat, n_feat) for i in range(2)],
        #     nn.Conv2d(n_feat, ms_channels, 3, padding=1, bias=False))
        
    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w] 
        # pan - high-resolution panchromatic image [N,1,H,W] 
        if type(pan) == torch.Tensor:
            pass
        elif pan==None:
            raise Exception('User does not provide pan image!')
        _,_,m,n = ms.shape
        _,_,M,N = pan.shape
        HR = upsample(ms, M, N)
        # HR = self.PGCU(pan, ms)

        # HR = self.ConvTrans1(ms)
        # HR = self.gnnmodel(HR, pan).to(device)
        # HR = self.Dysample(HR, pan).to(device)

        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)
            
        return HR

class GPPNN_GSCU(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN_GSCU, self).__init__()
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])
        self.GSCU = GaussianSplatter(kernel_size=5,c1=6,channels=ms_channels,n_feats= 48,n_classes=50)

    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        HR = self.GSCU(pan, ms)
        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)

        return HR

class GPPNN_PGCU(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN_PGCU, self).__init__()
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])
        self.PGCU = PGCU(ms_channels,128)

    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        HR = self.PGCU(pan, ms)
        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)

        return HR

class GPPNN_SFAU(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN_SFAU, self).__init__()
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])
        self.SFAU = SFAU(1,ms_channels)

    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        HR,_,_ = self.SFAU(ms,pan)
        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)

        return HR

class GPPNN_TConv(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN_TConv, self).__init__()
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])
        self.ConvTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1))

    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        HR = self.ConvTrans1(ms)
        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)

        return HR

class GPPNN_Nearest(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPPNN_Nearest, self).__init__()
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])

    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='nearest')

    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        HR = self.upsample(ms, M, N)
        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)

        return HR