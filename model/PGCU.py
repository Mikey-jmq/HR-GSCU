import torch
import torch.nn as nn
import torch.nn.functional as fun
from math import sqrt


class DownSamplingBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super(DownSamplingBlock, self).__init__()
        self.Conv = nn.Conv2d(in_channel, out_channel, (3,3), 2, 1)
        self.MaxPooling = nn.MaxPool2d((2, 2))
        
    def forward(self, x):
        out = self.MaxPooling(self.Conv(x))
        return out
    
class PGCU(nn.Module):
    
    def __init__(self, Channel=4, VecLen=128, NumberBlocks=3):
        super(PGCU, self).__init__()
        self.BandVecLen = VecLen//Channel
        self.Channel = Channel
        self.VecLen = VecLen
        
        ## Information Extraction
        # F.size == (Vec, W, H)
        self.FPConv = nn.Conv2d(1, Channel, (3,3), 1, 1)
        self.FMConv = nn.Conv2d(Channel, Channel, (3,3), 1, 1)
        self.FConv = nn.Conv2d(Channel*2, VecLen, (3,3), 1, 1)
        # G.size == (Vec, W/pow(2, N), H/pow(2, N))
        self.GPConv = nn.Sequential()
        self.GMConv = nn.Sequential()
        self.GConv = nn.Conv2d(Channel*2, VecLen, (3,3), 1, 1)
        for i in range(NumberBlocks):
            if i == 0:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, Channel))
            else:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(Channel, Channel))
                self.GMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(Channel, Channel))
        # V.size == (C, W/pow(2, N), H/pow(2, N)), k=W*H/64
        self.VPConv = nn.Sequential()
        self.VMConv = nn.Sequential()
        self.VConv = nn.Conv2d(Channel*2, Channel, (3,3), 1, 1)
        for i in range(NumberBlocks):
            if i == 0:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, Channel))
            else:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(Channel, Channel))
                self.VMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(Channel, Channel))

        # Linear Projection
        self.FLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.VecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.Channel)])
        self.GLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.VecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.Channel)])
        # FineAdjust
        self.FineAdjust = nn.Conv2d(Channel, Channel, (3,3), 1, 1)
        
    def forward(self, guide, x): #guide=pan, x=lrms
        up_x = fun.interpolate(x, scale_factor=(4,4), mode='nearest') #torch.Size([32, 4, 128, 128])
        Fm = self.FMConv(up_x) #torch.Size([32, 4, 128, 128])
        Fq = self.FPConv(guide) #torch.Size([32, 4, 128, 128])
        F = self.FConv(torch.cat([Fm, Fq], dim=1)) #torch.Size([32, 128, 128, 128])
        
        Gm = self.GMConv(x) #torch.Size([32, 4, 2, 2])
        Gp = self.GPConv(guide) #torch.Size([32, 4, 2, 2])
        G = self.GConv(torch.cat([Gm, Gp], dim=1)) #torch.Size([32, 128, 2, 2])
        
        Vm = self.VMConv(x) #torch.Size([32, 4, 2, 2])
        Vp = self.VPConv(guide) #torch.Size([32, 4, 2, 2])
        V = self.VConv(torch.cat([Vm, Vp], dim=1)) #torch.Size([32, 4, 2, 2])
        
        C = V.shape[1] #number of channel
        batch = G.shape[0] #batch size
        W, H = F.shape[2], F.shape[3] #Weight, Height
        OW, OH = G.shape[2], G.shape[3] #OW, OH: 2
        
        G = torch.transpose(torch.transpose(G, 1, 2), 2, 3) #torch.Size([32, 2, 2, 128])
        G = G.reshape(batch*OW*OH, self.VecLen) #torch.Size([128, 128])
        
        F = torch.transpose(torch.transpose(F, 1, 2), 2, 3) #torch.Size([32, 128, 128, 128])
        F = F.reshape(batch*W*H, self.VecLen) #torch.Size([524288, 128])
        BandsProbability = None
        for i in range(C):
            # F projection
            FVF = self.GLinear[i](G) #torch.Size([128, 32])
            FVF = FVF.reshape(batch, OW*OH, self.BandVecLen).transpose(-1, -2) # (batch, L, OW*OH) torch.Size([32, 32, 4])
            # G projection
            PVF = self.FLinear[i](F) #torch.Size([524288, 32])
            PVF = PVF.view(batch, W*H, self.BandVecLen) # (batch, W*H, L) torch.Size([32, 16384, 32])
            # Probability
            Probability = torch.bmm(PVF, FVF).reshape(batch*H*W, OW, OH) / sqrt(self.BandVecLen) #torch.Size([524288, 2, 2])
            Probability = torch.exp(Probability) / torch.sum(torch.exp(Probability), dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1) #torch.Size([524288, 2, 2])
            Probability = Probability.view(batch, W, H, 1, OW, OH) #torch.Size([32, 128, 128, 1, 2, 2])
            # Merge
            if BandsProbability is None:
                BandsProbability = Probability
            else:
                BandsProbability = torch.cat([BandsProbability, Probability], dim=3) #torch.Size([32, 128, 128, 4, 2, 2])
        #Information Entropy: H_map = torch.sum(BandsProbability*torch.log2(BandsProbability+1e-9), dim=(-1, -2, -3)) / C
        # Pro = BandsProbability[0,102:110,56:61,1,:,:]
        Pro = BandsProbability[0, 0:6, 107:112, 0, :, :]
        with open("tensor_out.txt", "w") as f:
            f.write(str(Pro) + "\n\n")

        out = torch.sum(BandsProbability*V.unsqueeze(dim=1).unsqueeze(dim=1), dim=(-1, -2))  #torch.Size([32, 128, 128, 4])
        out = out.transpose(-1, -2).transpose(1, 2) #torch.Size([32, 4, 128, 128])
        out = self.FineAdjust(out) #torch.Size([32, 4, 128, 128])
        # Out = out[0, :, 43:51, 0:5]
        Out = out[0, 0,  0:6, 107:112]
        with open("tensor_out_pixel.txt", "w") as f:
            f.write(str(Out) + "\n\n")
        return out
    

if __name__ == '__main__':
    PGCU = PGCU(4, 128)
    pan = torch.ones((32, 1, 128, 128))
    ms = torch.ones((32, 4, 32, 32))
    hrms = PGCU.forward(pan, ms)