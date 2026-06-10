import torch
import torch.nn as nn
import torch.nn.functional as fun
from model.BasicBlock import *
from model.PGCU import *
from model.gscu_lr import *
from model.dysamplenew import *
from model.SFAU import *

class PanNet(nn.Module):

    def __init__(self, channel, kernel_size=(3, 3), kernel_num=32):
        super(PanNet, self).__init__()
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            # ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size,
                                            #               kernel_num=kernel_num),

                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)

    def forward(self, pan, ms, hms, hpan):
        x_ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms

class PanNet_Nearest(nn.Module):

    def __init__(self, channel, kernel_size=(3, 3), kernel_num=32):
        super(PanNet_Nearest, self).__init__()
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            # ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size,
                                            #               kernel_num=kernel_num),

                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='nearest')

    def forward(self, pan, ms, hms, hpan):
        x_ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms

class PanNet_TConv(nn.Module):

    def __init__(self, channel, kernel_size=(3, 3), kernel_num=32):
        super(PanNet_TConv, self).__init__()
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            # ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size,
                                            #               kernel_num=kernel_num),

                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1,
                               output_padding=1),
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1,
                               output_padding=1))



    def forward(self, pan, ms, hms, hpan):
        x_ms = self.upsample(ms)
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms

class PanNet_PGCU(nn.Module):

    def __init__(self, channel, kernel_size=(3, 3), kernel_num=32):
        super(PanNet_PGCU, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.PGCU = PGCU(4, 128)
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            # ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size,
                                            #               kernel_num=kernel_num),

                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)

    def forward(self, pan, ms, hms, hpan):
        # x_ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        # up_ms = self.ConvTrans(hms)

        x_ms_new = self.PGCU(pan, ms)
        x = torch.cat([hpan, x_ms_new], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms_new

class PanNet_SFAU(nn.Module):

    def __init__(self, channel, k_up=5, kernel_size=(3, 3), kernel_num=32):
        super(PanNet_SFAU, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.SFAU = SFAU(y_channels=1, x_channels=channel, k_up=k_up) # ms, pan
        self.Conv1 = nn.Conv2d(in_channels=channel + 1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)

    def forward(self, pan, ms,hms, hpan):
        up_ms, _, _ = self.SFAU(ms, pan)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + up_ms


class PanNet_GSCU(nn.Module):

    def __init__(self, channel, kernel_size=(3, 3), kernel_num=32):
        super(PanNet_GSCU, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.GSCU = GaussianSplatter(kernel_size=5,c1=6,channels=channel,n_feats= 48,n_classes=50)
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            # ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size,
                                            #               kernel_num=kernel_num),

                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)

    def forward(self, pan, ms, hms, hpan):
        # x_ms = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        # up_ms = self.ConvTrans(hms)

        x_ms_new = self.GSCU(pan, ms)
        x = torch.cat([hpan, x_ms_new], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms_new

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hms = torch.rand(4,4,32,32).to(device)
    ms = torch.rand(4, 4, 32, 32).to(device)
    pan = torch.rand(4, 1, 128, 128).to(device)
    hpan = torch.rand(4, 1, 128, 128).to(device)
    PanNet_G = PanNet_GSCU(4).to(device)
    pred = PanNet_G.forward(pan, ms,hms, hpan)
    print("PanNet_GSCU输出维度为：", pred.shape)

    PanNet_P = PanNet_PGCU(4).to(device)
    pred = PanNet_P.forward(pan, ms,hms, hpan)
    print("PanNet_PGCU输出维度为：", pred.shape)

    PanNet = PanNet(channel=4).to(device)
    pred = PanNet.forward(pan, ms, hms, hpan)
    print("正常PanNet的输出维度为：", pred.shape)

    PanNet_n = PanNet_Nearest(4).to(device)
    pred = PanNet_n.forward(pan, ms, hms, hpan)
    print("PanNet_Nearest的输出维度为：", pred.shape)

    PanNet_T = PanNet_TConv(4).to(device)
    pred = PanNet_T.forward(pan, ms, hms, hpan)
    print("PanNet_Tconv的输出维度为：", pred.shape)