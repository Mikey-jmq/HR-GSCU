# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:34:49 2020

@author: win10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.PGCU import *
from model.gscu_final import *


class PNN(nn.Module):
    """Pansharpening by Convolutional Neural Networks (RS)
    doi.org/10.3390/rs8070594
    """
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat = 64):
        super(PNN, self).__init__()
        
        conv1 = nn.Conv2d(ms_channels+pan_channels, n_feat, 9, padding=4)
        conv2 = nn.Conv2d(n_feat, n_feat//2, 5, padding=2)
        conv3 = nn.Conv2d(n_feat//2, ms_channels, 5, padding=2)
        relu  = nn.ReLU(True)
        self.model = nn.Sequential(conv1, relu, conv2, relu, conv3)
        self.ConvTrans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1))


    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)
        
    def forward(self, ms, pan=None):
        if pan is None:
            raise Exception('User does not provide pan image!')
        ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])

        ms = torch.cat((ms, pan), dim=1)
        ms = self.model(ms)
        return ms


class PNN_Nearest(nn.Module):
    """Pansharpening by Convolutional Neural Networks (RS)
    doi.org/10.3390/rs8070594
    """

    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat=64):
        super(PNN_Nearest, self).__init__()

        conv1 = nn.Conv2d(ms_channels + pan_channels, n_feat, 9, padding=4)
        conv2 = nn.Conv2d(n_feat, n_feat // 2, 5, padding=2)
        conv3 = nn.Conv2d(n_feat // 2, ms_channels, 5, padding=2)
        relu = nn.ReLU(True)
        self.model = nn.Sequential(conv1, relu, conv2, relu, conv3)
        self.ConvTrans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1))

    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='nearest')

    def forward(self, ms, pan=None):
        if pan is None:
            raise Exception('User does not provide pan image!')
        ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        ms = torch.cat((ms, pan), dim=1)
        ms = self.model(ms)
        return ms

    class PNN(nn.Module):
        """Pansharpening by Convolutional Neural Networks (RS)
        doi.org/10.3390/rs8070594
        """

        def __init__(self,
                     ms_channels,
                     pan_channels,
                     n_feat=64):
            super(PNN, self).__init__()

            conv1 = nn.Conv2d(ms_channels + pan_channels, n_feat, 9, padding=4)
            conv2 = nn.Conv2d(n_feat, n_feat // 2, 5, padding=2)
            conv3 = nn.Conv2d(n_feat // 2, ms_channels, 5, padding=2)
            relu = nn.ReLU(True)
            self.model = nn.Sequential(conv1, relu, conv2, relu, conv3)
            # self.gnnmodel = adjGNN(ms_channels)
            # self.PGCU = PGCU(4, 128)
            # self.Dysample = DySample1(1)
            # self.Dy = DySample(4)
            self.ConvTrans = nn.Sequential(
                nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                                   padding=1, output_padding=1),
                nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                                   padding=1, output_padding=1))

        def upsample(self, x, h, w):
            return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

        def forward(self, ms, pan=None):
            if pan is None:
                raise Exception('User does not provide pan image!')
            ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
            # ms = self.PGCU(pan, ms)

            # x_ms = self.ConvTrans(ms)
            # ms = self.gnnmodel(x_ms, pan).to(device)
            # ms = self.Dysample(ms, pan).to(device)

            ms = torch.cat((ms, pan), dim=1)
            ms = self.model(ms)
            return ms


class PNN_Tconv(nn.Module):
    """Pansharpening by Convolutional Neural Networks (RS)
    doi.org/10.3390/rs8070594
    """

    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat=64):
        super(PNN_Tconv, self).__init__()

        conv1 = nn.Conv2d(ms_channels + pan_channels, n_feat, 9, padding=4)
        conv2 = nn.Conv2d(n_feat, n_feat // 2, 5, padding=2)
        conv3 = nn.Conv2d(n_feat // 2, ms_channels, 5, padding=2)
        relu = nn.ReLU(True)
        self.model = nn.Sequential(conv1, relu, conv2, relu, conv3)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1,
                               output_padding=1),
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1,
                               output_padding=1))
        self.ConvTrans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1))

    def forward(self, ms, pan=None):
        if pan is None:
            raise Exception('User does not provide pan image!')
        ms = self.upsample(ms)
        ms = torch.cat((ms, pan), dim=1)
        ms = self.model(ms)
        return ms


class PNN_PGCU(nn.Module):
    """Pansharpening by Convolutional Neural Networks (RS)
    doi.org/10.3390/rs8070594
    """

    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat=64):
        super(PNN_PGCU, self).__init__()

        conv1 = nn.Conv2d(ms_channels + pan_channels, n_feat, 9, padding=4)
        conv2 = nn.Conv2d(n_feat, n_feat // 2, 5, padding=2)
        conv3 = nn.Conv2d(n_feat // 2, ms_channels, 5, padding=2)
        relu = nn.ReLU(True)
        self.model = nn.Sequential(conv1, relu, conv2, relu, conv3)
        self.PGCU = PGCU(4, 128)
        self.ConvTrans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1))

    def forward(self, ms, pan=None):
        if pan is None:
            raise Exception('User does not provide pan image!')
        # ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        ms = self.PGCU(pan, ms)
        ms = torch.cat((ms, pan), dim=1)
        ms = self.model(ms)
        return ms

class PNN_GSCU(nn.Module):
    """Pansharpening by Convolutional Neural Networks (RS)
    doi.org/10.3390/rs8070594
    """

    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat=64):
        super(PNN_GSCU, self).__init__()

        conv1 = nn.Conv2d(ms_channels + pan_channels, n_feat, 9, padding=4)
        conv2 = nn.Conv2d(n_feat, n_feat // 2, 5, padding=2)
        conv3 = nn.Conv2d(n_feat // 2, ms_channels, 5, padding=2)
        relu = nn.ReLU(True)
        self.model = nn.Sequential(conv1, relu, conv2, relu, conv3)
        self.GSCU = GaussianSplatter(kernel_size=5,c1=6,channels=ms_channels,n_feats= 48,n_classes=50)
        self.ConvTrans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2,
                               padding=1, output_padding=1))

    def forward(self, ms, pan=None):
        if pan is None:
            raise Exception('User does not provide pan image!')
        # ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        ms = self.GSCU(pan, ms)
        ms = torch.cat((ms, pan), dim=1)
        ms = self.model(ms)
        return ms
# test
# from torchsummary import summary
# summary(PNN(10,1).cuda(), [(10,32,32),(1,64,64)])