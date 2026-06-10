# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:20:25 2020

@author: win10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN1 import *
from model.PGCU import *
from model.dysample import *

import sys
sys.path.append("..")
from utils1 import box_blur


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, padding=1)
        self.relu  = nn.ReLU(True)
        
    def forward(self, x):
        x = x+self.conv2(self.relu(self.conv1(x)))
        return x
    
class PANNET(nn.Module):
    """PanNet: A deep network architecture for pan-sharpening (ICCV)
    https://openaccess.thecvf.com/content_iccv_2017/html/Yang_PanNet_A_Deep_ICCV_2017_paper.html
    """

    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat = 32,
                 n_layer = 10):
        super(PANNET, self).__init__()
        
        relu = nn.ReLU(True)
        conv1 = nn.Conv2d(ms_channels+pan_channels, n_feat, 3, padding=1)
        res_blocks = [ResBlock(n_feat, n_feat) for _ in range(n_layer-2)]
        conv2 = nn.Conv2d(n_feat, ms_channels, 3, padding=1)
        
        self.model = nn.Sequential(
            conv1, 
            relu,
            *res_blocks,
            conv2
            )
        # self.PGCU = PGCU(4, 128)
        self.model = adjGNN(ms_channels)
        self.dysample = DySample(ms_channels)
        self.ConvTrans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1))
    
    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)
    
    def get_high_freq(self, x):
        return x-box_blur(x, kernel_size=[5,5])
    
    def forward(self, ms, pan=None):
        if pan is None:
            raise TypeError('Pan image must be supplied!')
        # get high frequency information
        x = torch.cat((self.upsample(self.get_high_freq(ms), pan.shape[-2], pan.shape[-1]),
                       self.get_high_freq(pan)),
                      dim=1)
        # ms_new = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        # ms_new = self.PGCU(pan, ms)

        x_ms = self.ConvTrans(ms)  # 16,4,128,128
        ms_new = self.model(x_ms, pan).to(device)

        ms = self.model(x, pan) + ms_new
        return ms
    
# test
# from torchsummary import summary
# summary(PANNET(10,1).cuda(), [(10,32,32),(1,64,64)])

# from torchstat import stat
# model = PANNET(3,3)
# stat(model, (6,265,256))

# import torch
# model = PANNET(3,3).cuda()
# A=torch.rand(1,3,256,256).cuda()
# B=torch.rand(1,3,256,256).cuda()
# from time import time
# t0 = time()
# a = model(A,B)
# print(a)
# print(time()-t0)