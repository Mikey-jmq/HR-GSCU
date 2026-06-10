import torch
import torch.nn as nn
import torch.optim as opt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import MyDataset, MyDataset1
from model.MSDCNN import *
from metrics_b import *
import os

# global config
device = 'cuda'
epoches = 100
batch_size_train = 6
batch_size_test = 1
ms_channels = 4
pan_channels = 1
# prepare dataset&dataloader
data_root = "d:/Users/Administrator/Desktop/GSCU/数据/GF2_data/GF2_data"
data_root1 = "d:/Users/Administrator/Desktop/GSCU/数据/test200_GF2"
train_pan = 'train128/pan'
train_ms = 'train128/ms'
test_pan = 'pan'
test_ms = 'ms'

train_dataset = MyDataset(data_root, train_ms, train_pan, 'bicubic')
test_dataset = MyDataset1(data_root1, test_ms, test_pan, 'bicubic')
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)  # 修改点：测试集没必要 shuffle，设为 False 即可

test_num = len(test_dataset)  # 动态获取测试集数量，替代写死的 num = 80

# PanNet_GSCU config
g_weight_decay = 5e-5
g_learning_rate = 2e-4
g_lossFun = nn.MSELoss()
GNNPanNet =  MSDCNN(ms_channels,pan_channels).to(device)

g_optimizer = opt.Adam(GNNPanNet.parameters(), lr=g_learning_rate, weight_decay=g_weight_decay)
scheduler_1 = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.5)
total = sum(p.numel() for p in GNNPanNet.parameters() if p.requires_grad)
print(f'total number of parameters: {total}')

# record trainning&testing
g_train_loss = []
best_qnr = -1.0
best_epoch = -1
b = []

os.makedirs('new_test/full', exist_ok=True)

# trainning
for epoch in tqdm(range(epoches)):
    g_loss = 0
    GNNPanNet.train()

    # ---------------- 训练阶段 (在降采样数据上进行) ----------------
    for label, pan, lrms, up_ms, hpan, hlrms in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        label = torch.Tensor(label).to(device).float()
        pan = torch.Tensor(pan).to(device).float()
        # hpan = torch.Tensor(hpan).to(device).float()
        lrms = torch.Tensor(lrms).to(device).float()
        # hlrms = torch.Tensor(hlrms).to(device).float()

        out = GNNPanNet.forward(lrms,pan)
        loss = g_lossFun(out, label)
        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()
        g_loss += loss.item()

    avg_train_loss = g_loss / len(train_loader)
    g_train_loss.append(avg_train_loss)
    print(f'Epoch: {epoch} | Train Loss (MSE): {avg_train_loss:.6f}')

    scheduler_1.step()

    # ---------------- 测试阶段 (在全尺度数据上验证 QNR) ----------------
    if epoch % 1 == 0:
        GNNPanNet.eval()

        QNR = 0
        DLAMBDA = 0
        DS = 0

        with torch.no_grad():
            for label, pan, lrms, hpan, hlrms in tqdm(test_loader, desc=f"Epoch {epoch} Evaluating"):
                pan = torch.Tensor(pan).to(device).float()
                lrms = torch.Tensor(lrms).to(device).float()
                # hpan = torch.Tensor(hpan).to(device).float()
                # hlrms = torch.Tensor(hlrms).to(device).float()

                out = GNNPanNet.forward(lrms,pan)

                out_np = out.permute(0, 2, 3, 1).cpu().numpy()
                lrms_np = lrms.permute(0, 2, 3, 1).cpu().numpy()
                pan_np = pan.permute(0, 2, 3, 1).cpu().numpy()

                batch_size_current = out_np.shape[0]

                # 遍历 Batch 计算指标
                for k in range(batch_size_current):
                    Out_img = out_np[k]  # shape: (H, W, C)
                    Lrms_img = lrms_np[k]  # shape: (H, W, C)
                    Pan_img = pan_np[k]  # shape: (H, W, C) 或者 (H, W, 1)

                    QNR += qnr(Out_img, Lrms_img, Pan_img)
                    DLAMBDA += D_lambda1(Out_img, Lrms_img)
                    DS += D_s1(Out_img, Lrms_img, Pan_img)

        # 计算当前 Epoch 的全尺度平均指标
        current_avg_qnr = QNR / test_num
        current_avg_dlambda = DLAMBDA / test_num
        current_avg_ds = DS / test_num

        print(f'Epoch: {epoch} | Full-Scale QNR: {current_avg_qnr:.4f} | D_lambda: {current_avg_dlambda:.4f} | D_s: {current_avg_ds:.4f}')

        # ---------------- 保存最佳模型 (早停机制) ----------------
        if current_avg_qnr > best_qnr:
            best_qnr = current_avg_qnr
            best_epoch = epoch

            # 只保存全尺度下 QNR 最高的那个权重
            best_filename = 'new_test/full/best_msdcnn1_full_gf2.pth'
            torch.save(GNNPanNet.state_dict(), best_filename)
            print(f'>>> New Best Model Saved at Epoch {epoch} with QNR: {best_qnr:.4f} <<<')

        # 如果 QNR 大于 0.80，记录下来（可选）
        # if current_avg_qnr > 0.80:
        #     b.append((epoch, current_avg_qnr, current_avg_dlambda, current_avg_ds))

print(f"=================================================")
print(f"训练结束！最高全尺度 QNR 为: {best_qnr:.6f}，在 epoch: {best_epoch} 取得。")
print(f"=================================================")