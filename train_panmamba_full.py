import torch
import torch.nn as nn
import torch.optim as opt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import MyDataset,MyDataset1
from utils.visualize import Evaluate
from utils.metrics import *
from model import get_sat_param
from panmamba.Mamba import PanMamba



# global config
# evaluater = Evaluate('GPPNN', 'GF2', device)
model_str = 'panmamba'
satellite_str = 'gf2'
# . Get the parameters of your satellite
sat_param = get_sat_param(satellite_str)
if sat_param!=None:
    ms_channels, pan_channels, scale = sat_param
else:
    print('You should specify `ms_channels`, `pan_channels` and `scale`! ')
    ms_channels = 4
    pan_channels = 1
    scale = 4

device = 'cuda:3'
epoches = 1000
batch_size = 8
num = 320
# prepare dataset&dataloader
data_root = "GF2_data1"
data_root1 = "all_test/test_GF2"
train_pan = 'train128/pan'
train_ms = 'train128/ms'
#test_pan = 'test128/pan'
#test_ms = 'test128/ms'
test_pan = 'pan'
test_ms = 'ms'
train_dataset = MyDataset(data_root, train_ms, train_pan, 'bicubic')
test_dataset = MyDataset1(data_root1, test_ms, test_pan, 'bicubic')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=4)

net = PanMamba().to(device)
#net.load_state_dict(torch.load('canshu/full/new/panmamba1.pth'))
#net.load_state_dict(torch.load('canshu/full/new/tc/panmamba_our_full15.pth'))
#net.load_state_dict(torch.load('canshu/full/new/panmamba_pgcu_full1.pth'))
net.load_state_dict(torch.load('canshu/full/mamba/panmamba_pgcu_full1.pth'))
# panmamba-GNN config
g_weight_decay = 5e-8
g_learning_rate = 5e-5
loss_fn = nn.MSELoss().to(device)
g_optimizer = opt.Adam(net.parameters(), lr=g_learning_rate, weight_decay=g_weight_decay)
scheduler_1 = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.1)
total = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'total number of parameters:{total}')
# record trainning&testing
g_train_loss = []
g_test_loss = []

# trainning
b = []
for epoch in tqdm(range(epoches)):
    g_loss = 0
    net.train()
    for label, pan, lrms, up_ms, hpan, hlrms in tqdm(train_loader):
       label = torch.Tensor(label).to(device).float()
       pan = torch.Tensor(pan).to(device).float()
       lrms = torch.Tensor(lrms).to(device).float()

       out = net.forward(lrms, pan)
       loss = loss_fn(out, label)
       g_optimizer.zero_grad()
       loss.backward()
       g_optimizer.step()
       g_loss += loss.item()

    g_train_loss.append(g_loss/train_loader.__len__())
    print('epoch:'+str(epoch),
          'panmamba_GNN train loss:'+str(g_loss/train_loader.__len__()))


    #torch.save(net.state_dict(), 'canshu/full/new/panmamba1.pth')
    
    # testing
    if epoch % 1 == 0:
        g_loss = 0
        net.eval()

        SAM = 0
        ERGAS = 0
        SSIM = 0
        SCC = 0
        PSNR = 0

        QNR = 0
        DLAMBDA = 0
        DS = 0

        for label, pan, lrms, hpan, hlrms in tqdm(test_loader):
        #for label, pan, lrms, up_ms, hpan, hlrms in tqdm(test_loader):
            #label = torch.Tensor(label).to(device).float()
            pan = torch.Tensor(pan).to(device).float()
            #lrms = torch.Tensor(lrms).to(device).float()
            lrms = torch.Tensor(lrms.permute(0,2,1,3)).to(device).float()
            out = net.forward(lrms, pan)
            #loss1 = loss_fn(out, label)
            out1 = out.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            lrms = lrms.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            #label = label.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            pan = pan.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            #g_loss += loss1.item()

            Sam = 0
            Ergas = 0
            Ssim = 0
            Scc = 0
            Psnr = 0

            Qnr = 0
            Dlambda = 0
            Ds = 0

            i = out1.shape[0]
            for k in range(i):
                Out = out1[k]
                Lrms = lrms[k]
                Label = label[k]
                Pan = pan[k]

                #Sam += sam(Out, Label)
                #Ergas += ergas(Out, Label)
                #Ssim += ssim(Out, Label)
                #Scc += scc(Out, Label)
                #Psnr += psnr(Out, Label)

                Out = Out.squeeze(0).cpu().detach().numpy()
                Lrms = Lrms.squeeze(0).cpu().detach().numpy()
                Pan= Pan.squeeze(0).cpu().detach().numpy()
                Qnr += qnr(Out, Lrms, Pan)
                Dlambda += D_lambda(Out, Lrms)
                Ds += D_s(Out, Lrms,Pan)

            SAM += Sam
            ERGAS += Ergas
            SSIM += Ssim
            SCC += Scc
            PSNR += Psnr

            QNR += Qnr
            DLAMBDA += Dlambda
            DS += Ds


        g_test_loss.append(g_loss/test_loader.__len__())

        print('epoch:'+str(epoch),
              'panmamba_GNN test loss:'+str(g_loss/test_loader.__len__()),
              '-',
              'QNR:' + str(QNR / num),
              'DLAMBDA:' + str(DLAMBDA / num),
              'DS:' + str(DS / num)
              )

        print('epoch:'+str(epoch),
              'panmamba_GNN test loss:'+str(g_loss/test_loader.__len__()),
              '\n',
              'SAM:' + str(SAM / num),
              'ERGAS:' + str(ERGAS / num),
              'SSIM:' + str(SSIM / num),
              'SCC:' + str(SCC / num),
              'PSNR:' + str(PSNR / num)
              )
        #if PSNR/num > 48.9:
            #b.append(epoch)
            #b.append(SAM/num)
            #b.append(ERGAS/num)
            #b.append(SSIM/num)
            #b.append(SCC/num)
            #b.append(PSNR/num)
        #if QNR / num > 0.60:
         #   filename = f'canshu/full/new/panmamba_full_gf2{epoch}.pth'
          #  torch.save(net.state_dict(), filename)

        if QNR / num > 0.834 and DS / num < 0.0025:
            b.append(epoch)
            b.append(QNR / num)
            b.append(DLAMBDA / num)
            b.append(DS / num)
            filename = f'canshu/full/mamba/ori/panmamba_ours_full{epoch}.pth'
            torch.save(net.state_dict(), filename)
print(b)