import torch
import torch.nn as nn
import torch.optim as opt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import MyDataset
from utils.visualize import Evaluate
from utils.metrics import *
from model import get_sat_param
from model.MSDCNN import *
import time


# global config
evaluater = Evaluate('MSDCNN_GSCU', 'GF2', device)
model_str = 'MSDCNN_GSCU'
satellite_str = 'GF2'
# . Get the parameters of your satellite
sat_param = get_sat_param(satellite_str)
if sat_param!=None:
    ms_channels, pan_channels, scale = sat_param
else:
    print('You should specify `ms_channels`, `pan_channels` and `scale`! ')
    ms_channels = 4
    pan_channels = 1
    scale = 4

device = 'cuda:0'
epoches = 300
batch_size = 16
# prepare dataset&dataloader
data_root = "d:/Users/Administrator/Desktop/GSCU/数据/GF2_data/GF2_data"
train_pan = 'train128/pan'
train_ms = 'train128/ms'
test_pan = 'test128/pan'
test_ms = 'test128/ms'
# test_pan = 'test/pan'
# test_ms = 'test/ms'
train_dataset = MyDataset(data_root, train_ms, train_pan, 'bicubic')
test_dataset = MyDataset(data_root, test_ms, test_pan, 'bicubic')
num = len(test_dataset)
print(f"训练集数量: {len(train_dataset)}, 测试集数量: {num}")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

Model = MSDCNN_GSCU(ms_channels,pan_channels).to(device)

g_weight_decay = 5e-8
g_learning_rate = 1e-4
g_lossFun = nn.L1Loss().to(device)
g_optimizer = opt.Adam(Model.parameters(), lr=g_learning_rate, weight_decay=g_weight_decay)
# scheduler_1 = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.1)
total = sum(p.numel() for p in Model.parameters() if p.requires_grad)
print(f'total number of parameters:{total}')

# record trainning&testing
g_train_loss = []
g_test_loss = []
best_psnr = 0.0

# trainning
b = []
for epoch in tqdm(range(epoches)):
    epoch_train_loss = 0.0
    Model.train()
    for label, pan, lrms, up_ms, hpan, hlrms in tqdm(train_loader):
        label = torch.Tensor(label).to(device).float()
        pan = torch.Tensor(pan).to(device).float()
        lrms = torch.Tensor(lrms).to(device).float()

        out = Model.forward(lrms,pan)
        loss = g_lossFun(out, label)
        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()
        epoch_train_loss += loss.item()

    # scheduler.step()
    avg_train_loss = epoch_train_loss / len(train_loader)#记录损失曲线的
    g_train_loss.append(avg_train_loss)
    print(f'Epoch: {epoch} | Train Loss: {avg_train_loss:.6f} | LR: {g_optimizer.param_groups[0]["lr"]:.6f}')

    # testing
    if epoch % 1 == 0:
        epoch_test_loss = 0.0
        Model.eval()

        total_time = 0.0
        metrics_sum = {'SAM': 0, 'ERGAS': 0, 'SSIM': 0, 'SCC': 0, 'PSNR': 0}

        for label, pan, lrms, up_ms, hpan, hlrms in tqdm(test_loader):
        # for label, pan, lrms, hpan, hlrms in tqdm(test_loader):
            label = torch.Tensor(label).to(device).float()
            pan = torch.Tensor(pan).to(device).float()
            lrms = torch.Tensor(lrms).to(device).float()
            # lrms = torch.Tensor(lrms.permute(0,2,1,3)).to(device).float()

            torch.cuda.synchronize()
            start_time = time.time()
            out = Model.forward(lrms, pan)
            torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)

            loss_t = g_lossFun(out, label)
            epoch_test_loss += loss_t.item()
            out = out.detach()
            label = label.detach()
            out_perm = out.permute(0, 2, 3, 1)
            label_perm = label.permute(0, 2, 3, 1)
            batch_sz = out_perm.shape[0]

            for k in range(batch_sz):
            # 单张图片计算
                img_pred = out_perm[k]
                img_gt = label_perm[k]

            # 累加指标 (请确保 imported 的 metrics 函数能处理 (H,W,C) 格式)
                metrics_sum['SAM'] += sam(img_pred, img_gt)
                metrics_sum['ERGAS'] += ergas(img_pred, img_gt)
                metrics_sum['SSIM'] += ssim(img_pred, img_gt)
                metrics_sum['SCC'] += scc(img_pred, img_gt)
                metrics_sum['PSNR'] += psnr(img_pred, img_gt)


        avg_test_loss = epoch_test_loss / len(test_loader)
        g_test_loss.append(avg_test_loss)

        avg_metrics = {k: v / num for k, v in metrics_sum.items()}

        print(f"Epoch {epoch} Test Metrics: "
              f"PSNR: {avg_metrics['PSNR']:.4f} | "
              f"SSIM: {avg_metrics['SSIM']:.4f} | "
              f"SAM: {avg_metrics['SAM']:.4f}")

        if avg_metrics['PSNR'] > best_psnr:
            best_psnr = avg_metrics['PSNR']
            # 调用 evaluator 保存
            evaluater.save_best(Model, epoch, avg_metrics)

            # 每 10 轮更新一次 loss 曲线图
        if epoch % 10 == 0:
            evaluater.visualize(g_train_loss, g_test_loss)
