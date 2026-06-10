# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:27:14 2020

@author: win10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN1 import *
from model.PGCU import *
from model.dysample1 import *
from model.dysample import *
from model.gscu_final import *
from model.SFAU import *

class MSDCNN(nn.Module):
    """A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening (JSTARS)
    doi.org/10.1109/JSTARS.2018.2794888
    """

    def __init__(self,
                 ms_channels,
                 pan_channels):
        super().__init__()
        self.conv1   = nn.Conv2d(ms_channels+pan_channels,60,kernel_size=7,padding=3)
        self.conv2_1 = nn.Conv2d(60,20,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(60,20,kernel_size=5,padding=2)
        self.conv2_3 = nn.Conv2d(60,20,kernel_size=7,padding=3)
        self.conv3   = nn.Conv2d(60,30,kernel_size=3,padding=1)
        self.conv4_1 = nn.Conv2d(30,10,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(30,10,kernel_size=5,padding=2)
        self.conv4_3 = nn.Conv2d(30,10,kernel_size=7,padding=3)
        self.conv5   = nn.Conv2d(30,ms_channels,kernel_size=5,padding=2)
        self.conv6   = nn.Conv2d(ms_channels+pan_channels,64,kernel_size=9,padding=4)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(32, ms_channels, kernel_size=5, padding=2)

        # self.PGCU = PGCU(4, 128)

    def cnn_deep(self, x):
        x1 = F.relu(self.conv1(x))
        ms1_1 = F.relu(self.conv2_1(x1))
        ms1_2 = F.relu(self.conv2_2(x1))
        ms1_3 = F.relu(self.conv2_3(x1))
        x2 = x1 + torch.cat([ms1_1, ms1_2, ms1_3], dim=1)
        x3 = F.relu(self.conv3(x2))
        ms2_1 = F.relu(self.conv4_1(x3))
        ms2_2 = F.relu(self.conv4_2(x3))
        ms2_3 = F.relu(self.conv4_3(x3))
        x4 = x3 + torch.cat([ms2_1, ms2_2, ms2_3], dim=1)
        out_deep = self.conv5(x4)
        return out_deep

    def cnn_shallow(self, x):
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out_shallow = self.conv8(x)
        return out_shallow

    def forward(self, ms, pan=None):
        if pan is None:
            raise TypeError('Pan image must be supplied!')
        ms_up = F.interpolate(ms, size=pan.shape[-2:], mode='bicubic', align_corners=True)
        stacked_input = torch.cat((ms_up, pan), dim=1)
        out = self.cnn_deep(stacked_input) + self.cnn_shallow(stacked_input)

        return out

class MSDCNN_TConv(nn.Module):
    """A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening (JSTARS)
    doi.org/10.1109/JSTARS.2018.2794888
    """

    def __init__(self,
                 ms_channels,
                 pan_channels):
        super().__init__()
        self.conv1   = nn.Conv2d(ms_channels+pan_channels,60,kernel_size=7,padding=3)
        self.conv2_1 = nn.Conv2d(60,20,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(60,20,kernel_size=5,padding=2)
        self.conv2_3 = nn.Conv2d(60,20,kernel_size=7,padding=3)
        self.conv3   = nn.Conv2d(60,30,kernel_size=3,padding=1)
        self.conv4_1 = nn.Conv2d(30,10,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(30,10,kernel_size=5,padding=2)
        self.conv4_3 = nn.Conv2d(30,10,kernel_size=7,padding=3)
        self.conv5   = nn.Conv2d(30,ms_channels,kernel_size=5,padding=2)
        self.conv6   = nn.Conv2d(ms_channels+pan_channels,64,kernel_size=9,padding=4)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(32, ms_channels, kernel_size=5, padding=2)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2, padding=1,
                               output_padding=1),
            nn.ConvTranspose2d(in_channels=ms_channels, out_channels=ms_channels, kernel_size=(3, 3), stride=2, padding=1,
                               output_padding=1))

        # self.PGCU = PGCU(4, 128)

    def cnn_deep(self, x):
        x1 = F.relu(self.conv1(x))
        ms1_1 = F.relu(self.conv2_1(x1))
        ms1_2 = F.relu(self.conv2_2(x1))
        ms1_3 = F.relu(self.conv2_3(x1))
        x2 = x1 + torch.cat([ms1_1, ms1_2, ms1_3], dim=1)
        x3 = F.relu(self.conv3(x2))
        ms2_1 = F.relu(self.conv4_1(x3))
        ms2_2 = F.relu(self.conv4_2(x3))
        ms2_3 = F.relu(self.conv4_3(x3))
        x4 = x3 + torch.cat([ms2_1, ms2_2, ms2_3], dim=1)
        out_deep = self.conv5(x4)
        return out_deep

    def cnn_shallow(self, x):
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out_shallow = self.conv8(x)
        return out_shallow

    def forward(self, ms, pan=None):
        if pan is None:
            raise TypeError('Pan image must be supplied!')
        ms_up = self.upsample(ms)
        stacked_input = torch.cat((ms_up, pan), dim=1)
        out = self.cnn_deep(stacked_input) + self.cnn_shallow(stacked_input)

        return out

class MSDCNN_Nearest(nn.Module):
    """A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening (JSTARS)
    doi.org/10.1109/JSTARS.2018.2794888
    """

    def __init__(self,
                 ms_channels,
                 pan_channels):
        super().__init__()
        self.conv1   = nn.Conv2d(ms_channels+pan_channels,60,kernel_size=7,padding=3)
        self.conv2_1 = nn.Conv2d(60,20,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(60,20,kernel_size=5,padding=2)
        self.conv2_3 = nn.Conv2d(60,20,kernel_size=7,padding=3)
        self.conv3   = nn.Conv2d(60,30,kernel_size=3,padding=1)
        self.conv4_1 = nn.Conv2d(30,10,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(30,10,kernel_size=5,padding=2)
        self.conv4_3 = nn.Conv2d(30,10,kernel_size=7,padding=3)
        self.conv5   = nn.Conv2d(30,ms_channels,kernel_size=5,padding=2)
        self.conv6   = nn.Conv2d(ms_channels+pan_channels,64,kernel_size=9,padding=4)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(32, ms_channels, kernel_size=5, padding=2)

        # self.PGCU = PGCU(4, 128)

    def cnn_deep(self, x):
        x1 = F.relu(self.conv1(x))
        ms1_1 = F.relu(self.conv2_1(x1))
        ms1_2 = F.relu(self.conv2_2(x1))
        ms1_3 = F.relu(self.conv2_3(x1))
        x2 = x1 + torch.cat([ms1_1, ms1_2, ms1_3], dim=1)
        x3 = F.relu(self.conv3(x2))
        ms2_1 = F.relu(self.conv4_1(x3))
        ms2_2 = F.relu(self.conv4_2(x3))
        ms2_3 = F.relu(self.conv4_3(x3))
        x4 = x3 + torch.cat([ms2_1, ms2_2, ms2_3], dim=1)
        out_deep = self.conv5(x4)
        return out_deep

    def cnn_shallow(self, x):
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out_shallow = self.conv8(x)
        return out_shallow

    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='nearest')

    def forward(self, ms, pan=None):
        if pan is None:
            raise TypeError('Pan image must be supplied!')
        ms_up = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        stacked_input = torch.cat((ms_up, pan), dim=1)
        out = self.cnn_deep(stacked_input) + self.cnn_shallow(stacked_input)

        return out

class MSDCNN_PGCU(nn.Module):
    """A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening (JSTARS)
    doi.org/10.1109/JSTARS.2018.2794888
    """

    def __init__(self,
                 ms_channels,
                 pan_channels):
        super().__init__()
        self.conv1   = nn.Conv2d(ms_channels+pan_channels,60,kernel_size=7,padding=3)
        self.conv2_1 = nn.Conv2d(60,20,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(60,20,kernel_size=5,padding=2)
        self.conv2_3 = nn.Conv2d(60,20,kernel_size=7,padding=3)
        self.conv3   = nn.Conv2d(60,30,kernel_size=3,padding=1)
        self.conv4_1 = nn.Conv2d(30,10,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(30,10,kernel_size=5,padding=2)
        self.conv4_3 = nn.Conv2d(30,10,kernel_size=7,padding=3)
        self.conv5   = nn.Conv2d(30,ms_channels,kernel_size=5,padding=2)
        self.conv6   = nn.Conv2d(ms_channels+pan_channels,64,kernel_size=9,padding=4)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(32, ms_channels, kernel_size=5, padding=2)

        self.PGCU = PGCU(4, 128)

    def cnn_deep(self, x):
        x1 = F.relu(self.conv1(x))
        ms1_1 = F.relu(self.conv2_1(x1))
        ms1_2 = F.relu(self.conv2_2(x1))
        ms1_3 = F.relu(self.conv2_3(x1))
        x2 = x1 + torch.cat([ms1_1, ms1_2, ms1_3], dim=1)
        x3 = F.relu(self.conv3(x2))
        ms2_1 = F.relu(self.conv4_1(x3))
        ms2_2 = F.relu(self.conv4_2(x3))
        ms2_3 = F.relu(self.conv4_3(x3))
        x4 = x3 + torch.cat([ms2_1, ms2_2, ms2_3], dim=1)
        out_deep = self.conv5(x4)
        return out_deep

    def cnn_shallow(self, x):
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out_shallow = self.conv8(x)
        return out_shallow

    def forward(self, ms, pan=None):
        if pan is None:
            raise TypeError('Pan image must be supplied!')
        ms_up = self.PGCU(pan, ms)
        stacked_input = torch.cat((ms_up, pan), dim=1)
        out = self.cnn_deep(stacked_input) + self.cnn_shallow(stacked_input)

        return out


class MSDCNN_GSCU(nn.Module):
    """A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening (JSTARS)
    doi.org/10.1109/JSTARS.2018.2794888
    """

    def __init__(self,
                 ms_channels,
                 pan_channels):
        super().__init__()
        self.conv1   = nn.Conv2d(ms_channels+pan_channels,60,kernel_size=7,padding=3)
        self.conv2_1 = nn.Conv2d(60,20,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(60,20,kernel_size=5,padding=2)
        self.conv2_3 = nn.Conv2d(60,20,kernel_size=7,padding=3)
        self.conv3   = nn.Conv2d(60,30,kernel_size=3,padding=1)
        self.conv4_1 = nn.Conv2d(30,10,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(30,10,kernel_size=5,padding=2)
        self.conv4_3 = nn.Conv2d(30,10,kernel_size=7,padding=3)
        self.conv5   = nn.Conv2d(30,ms_channels,kernel_size=5,padding=2)
        self.conv6   = nn.Conv2d(ms_channels+pan_channels,64,kernel_size=9,padding=4)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(32, ms_channels, kernel_size=5, padding=2)

        self.GSCU = GaussianSplatter(kernel_size=5,c1=6,channels=ms_channels,n_feats= 48,n_classes=50)

    def cnn_deep(self, x):
        x1 = F.relu(self.conv1(x))
        ms1_1 = F.relu(self.conv2_1(x1))
        ms1_2 = F.relu(self.conv2_2(x1))
        ms1_3 = F.relu(self.conv2_3(x1))
        x2 = x1 + torch.cat([ms1_1, ms1_2, ms1_3], dim=1)
        x3 = F.relu(self.conv3(x2))
        ms2_1 = F.relu(self.conv4_1(x3))
        ms2_2 = F.relu(self.conv4_2(x3))
        ms2_3 = F.relu(self.conv4_3(x3))
        x4 = x3 + torch.cat([ms2_1, ms2_2, ms2_3], dim=1)
        out_deep = self.conv5(x4)
        return out_deep

    def cnn_shallow(self, x):
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out_shallow = self.conv8(x)
        return out_shallow

    def forward(self, ms, pan=None):
        if pan is None:
            raise TypeError('Pan image must be supplied!')
        ms_up = self.GSCU(pan, ms)
        stacked_input = torch.cat((ms_up, pan), dim=1)
        out = self.cnn_deep(stacked_input) + self.cnn_shallow(stacked_input)

        return out


class MSDCNN_SFAU(nn.Module):
    """A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening (JSTARS)
    doi.org/10.1109/JSTARS.2018.2794888
    """

    def __init__(self,
                 ms_channels,
                 pan_channels):
        super().__init__()
        self.conv1   = nn.Conv2d(ms_channels+pan_channels,60,kernel_size=7,padding=3)
        self.conv2_1 = nn.Conv2d(60,20,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(60,20,kernel_size=5,padding=2)
        self.conv2_3 = nn.Conv2d(60,20,kernel_size=7,padding=3)
        self.conv3   = nn.Conv2d(60,30,kernel_size=3,padding=1)
        self.conv4_1 = nn.Conv2d(30,10,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(30,10,kernel_size=5,padding=2)
        self.conv4_3 = nn.Conv2d(30,10,kernel_size=7,padding=3)
        self.conv5   = nn.Conv2d(30,ms_channels,kernel_size=5,padding=2)
        self.conv6   = nn.Conv2d(ms_channels+pan_channels,64,kernel_size=9,padding=4)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(32, ms_channels, kernel_size=5, padding=2)

        self.SFAU = SFAU(1,ms_channels)

    def cnn_deep(self, x):
        x1 = F.relu(self.conv1(x))
        ms1_1 = F.relu(self.conv2_1(x1))
        ms1_2 = F.relu(self.conv2_2(x1))
        ms1_3 = F.relu(self.conv2_3(x1))
        x2 = x1 + torch.cat([ms1_1, ms1_2, ms1_3], dim=1)
        x3 = F.relu(self.conv3(x2))
        ms2_1 = F.relu(self.conv4_1(x3))
        ms2_2 = F.relu(self.conv4_2(x3))
        ms2_3 = F.relu(self.conv4_3(x3))
        x4 = x3 + torch.cat([ms2_1, ms2_2, ms2_3], dim=1)
        out_deep = self.conv5(x4)
        return out_deep

    def cnn_shallow(self, x):
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out_shallow = self.conv8(x)
        return out_shallow

    def forward(self, ms, pan=None):
        if pan is None:
            raise TypeError('Pan image must be supplied!')
        ms_up,_,_ = self.SFAU(ms,pan)
        stacked_input = torch.cat((ms_up, pan), dim=1)
        out = self.cnn_deep(stacked_input) + self.cnn_shallow(stacked_input)

        return out