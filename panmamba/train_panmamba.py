import torch

import torch.nn.functional as fun
import torch.nn as nn
import torch.optim as opt
from metrics_b import *
from metrics import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MyDataset
from visualize import Evaluate
from Mamba import PanMamba, PanMamba_PGCU, PanMamba_swin
from time import time

torch.cuda.empty_cache()

# global config
device = 'cuda'
epoches = 1001
batch_size = 4
evaluater = Evaluate('GPPNN', 'WV3', device)
# prepare dataset&dataloader
data_root = '/home/sk/pytorch/WV3_data'
train_pan = 'train128/pan'
train_ms = 'train128/ms'

test_pan = 'test128/pan'
test_ms = 'test128/ms'

# test_pan = 'test4/pan'
# test_ms = 'test4/ms'

train_dataset = MyDataset(data_root, train_ms, train_pan, 'bicubic')
test_dataset = MyDataset(data_root, test_ms, test_pan, 'bicubic')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#GPPNN config
g_weight_decay = 1e-8
g_learning_rate = 1e-3
g_lossFun = nn.MSELoss()
# GPPNN = GPPNN(4,1,64,8).to(device)
PanMamba = PanMamba().to(device)
g_optimizer = opt.Adam(PanMamba.parameters(), lr=g_learning_rate, weight_decay=g_weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.1)

#GPPNN_PGCU config
p_weight_decay = 1e-8
p_learning_rate = 1e-3
p_lossFun = nn.MSELoss()
PanMamba_PGCU = PanMamba_PGCU().to(device)
p_optimizer = opt.Adam(PanMamba_PGCU.parameters(), lr=p_learning_rate, weight_decay=p_weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(p_optimizer, step_size=20, gamma=0.01)

#GPPNN_swin config
t_weight_decay = 1e-8
t_learning_rate = 5e-4
t_lossFun = nn.MSELoss()
PanMamba_swin = PanMamba_swin().to(device)
t_optimizer = opt.Adam(PanMamba_swin.parameters(), lr=t_learning_rate, weight_decay=t_weight_decay)
scheduler_1 = torch.optim.lr_scheduler.StepLR(t_optimizer, step_size=20, gamma=0.1)


# record trainning&testing

g_train_loss = []
g_test_loss = []

p_train_loss = []
p_test_loss = []

t_train_loss = []
t_test_loss = []

torch.cuda.empty_cache()

# trainning
for epoch in tqdm(range(epoches)):
    # trainning

    g_loss = 0
    PanMamba.train()

    p_loss = 0
    PanMamba_PGCU.train()

    t_loss = 0
    PanMamba_swin.train()

    for label, pan, lrms, up_ms, hpan, hlrms in tqdm(train_loader):
        label = torch.Tensor(label).to(device).float()
        pan = torch.Tensor(pan).to(device).float()
        hpan = torch.Tensor(hpan).to(device).float()
        lrms = torch.Tensor(lrms).to(device).float()
        hlrms = torch.Tensor(hlrms).to(device).float()

        # GPPNN
        # out = PanMamba.forward(lrms, pan)
        # loss_1 = g_lossFun(out, label)
        # # optional: for residual structure
        # loss = loss_1
        # g_optimizer.zero_grad()
        # loss.backward()
        # g_optimizer.step()
        # g_loss += loss_1.item()

        #GPPNN_PGCU
        # out = PanMamba_PGCU.forward(lrms, pan)
        # loss_2 = p_lossFun(out, label)
        # # optional: for residual structure
        # loss = loss_2
        # p_optimizer.zero_grad()
        # loss.backward()
        # p_optimizer.step()
        # p_loss += loss_2.item()
        #
        #
        # # #PanNet_swin
        out = PanMamba_swin.forward(lrms, pan)
        loss_3 = t_lossFun(out, label)
        # optional: for residual structure
        loss = loss_3
        t_optimizer.zero_grad()
        loss.backward()
        t_optimizer.step()
        t_loss += loss_3.item()

    g_train_loss.append(g_loss / train_loader.__len__())
    p_train_loss.append(p_loss / train_loader.__len__())
    t_train_loss.append(t_loss/train_loader.__len__())

    print('epoch:'+str(epoch),
          # 'PanMamba train loss:' + str(g_loss / train_loader.__len__()),
          # 'PanMamba_PGCU train loss:' + str(p_loss / train_loader.__len__()),
          'PanMamba_swin train loss:' + str(t_loss / train_loader.__len__()),
          )

    torch.cuda.empty_cache()

    # testing
    if epoch%1 == 0:

        g_loss = 0
        PanMamba.eval()

        p_loss = 0
        PanMamba_PGCU.eval()

        t_loss = 0
        PanMamba_swin.eval()

        SAM = 0
        ERGAS = 0
        SCC = 0
        SSIM = 0
        PSNR = 0
        QNr = 0
        D_Lambda = 0
        D_S = 0
        NUM = 0

        if epoch == 0:
            time_sum = 0

        time_start = time()
        for label, pan, lrms, up_ms, hpan, hlrms in tqdm(test_loader):
            label = label.to(device).float()
            pan = pan.to(device).float()
            hpan = hpan.to(device).float()
            lrms = lrms.to(device).float()
            hlrms = hlrms.to(device).float()
            LABEL = label
            Lrms = lrms.permute(0, 3, 2, 1).permute(0, 2, 1, 3)

            # PanMamba
            # out = PanMamba.forward(lrms,pan)
            # loss = g_lossFun(out, label)
            # out1 = out.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            # label = label.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            # g_loss += loss.item()

            #PanMamba_PGCU
            # out = PanMamba_PGCU.forward(lrms,pan)
            # loss = p_lossFun(out, label)
            # out1 = out.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            # label = label.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            # p_loss += loss.item()
            #
            #
            # #PanMamba_swin
            out = PanMamba_swin.forward(lrms,pan)
            loss = t_lossFun(out, label)
            out1 = out.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            label = label.permute(0, 3, 2, 1).permute(0, 2, 1, 3)
            t_loss += loss.item()

            Sam = 0
            Ergas = 0
            Scc = 0
            Ssim = 0
            Psnr = 0
            Qnr = 0
            D_lambda = 0
            D_s = 0
            i = out1.shape[0]
            Num = 0
            Pan = pan.permute(0, 3, 2, 1).permute(0, 2, 1, 3)

            for k in range(i):
                Out = out1[k]
                Label = label[k]
                Lrms1 = Lrms[k]
                Pan1 = Pan[k]

                Sam += sam(Out, Label)
                Ergas += ergas(Out, Label)
                Scc += scc(Out, Label)
                Ssim += ssim(Out, Label)
                Psnr += psnr(Out, Label)
                Out = Out.squeeze(0).detach().cpu().numpy()
                LRms = Lrms1.detach().cpu().numpy()
                PAn = Pan1.detach().cpu().numpy()

                # Qnr += qnr(Out, LRms, PAn)
                # D_lambda += D_lambda1(Out, LRms)
                # D_s += D_s1(Out, LRms, PAn)
                Num += 1

            SAM += Sam
            ERGAS += Ergas
            SCC += Scc
            SSIM += Ssim
            PSNR += Psnr
            # QNr += Qnr
            # D_Lambda += D_lambda
            # D_S += D_s
            NUM += Num


        t_test_loss.append(t_loss/test_loader.__len__())



        print('epoch:'+str(epoch),
              # 'PanMamba test loss:' + str(g_loss / test_loader.__len__()),
              # 'PanMamba_PGCU test loss:' + str(p_loss / test_loader.__len__()),
              'PanMamba_swin test loss:'+str(t_loss/test_loader.__len__()),
              # 'SAM:'+str(SAM/80),
              # 'ERGAS:'+str(ERGAS/80),
              # 'SSIM:'+str(SSIM/80),
              # 'SCC:'+str(SCC/80),
              # 'PSNR:'+str(PSNR/80),
              # 'QINDEX:'+str(QINDEX/80),

              'SAM:' + str(SAM / NUM),
              'ERGAS:' + str(ERGAS / NUM),
              'SSIM:' + str(SSIM / NUM),
              'SCC:' + str(SCC / NUM),
              'PSNR:' + str(PSNR / NUM),
              # 'D_Lambda:' + str(D_Lambda / NUM),
              #  'D_S:' + str(D_S / NUM),

              )
# print(time_sum/10)
torch.cuda.empty_cache()
