# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import os

#
# class Evaluate():
#
#     def __init__(self, exp_type, data_type, device):
#         self.path = 'result/'+exp_type+'/'+data_type+'exp'
#         self.data_type = data_type
#         self.exp_type = exp_type
#         self.device = device
#         self.best_o = 1
#         self.best_g = 1
#         i = 0
#         while True:
#             folder = os.path.exists(self.path + str(i))
#             if folder is False:
#                 self.path = self.path + str(i) + '/'
#                 os.makedirs(self.path)
#                 break
#             i += 1
#
#     def visualize(self, g_train, g_test, MSDCNN_gcn):
#         # Save the best model on training datasets
#         # if self.best_o > o_train[-1]:
#         #     self.best_o = o_train[-1]
#         #     torch.save(pannet, self.path+self.exp_type+'.pkl')
#         # if self.best_g > o_train[-1]:
#         #     self.best_g = o_train[-1]
#         #     torch.save(pannet_gcn, self.path+self.exp_type+'_gcn.pkl')
#
#         ##3text_o = self.exp_type
#         test_g = self.exp_type + ' with GNN'
#
#         index = np.arange(len(g_train))
#         index_ = np.arange(0, len(g_train), 1)
#
#         plt.figure(1)
#         plt.grid(color='#7d7f7c', linestyle='-.')
#         # plt.plot(index, o_train, 'c', linewidth=1.5, label="train "+text_o)
#         # plt.plot(index_, o_test, '2c--', linewidth=1.5, label="test "+text_o)
#         plt.plot(index, g_train, 'r', linewidth=1.5, label="train "+test_g)
#         plt.plot(index_, g_test, '2r--', linewidth=1.5, label="test "+test_g)
#         plt.title('Loss:PanNetwithGNN')
#         plt.xlabel('epoch')
#         plt.ylabel('MSE')
#         plt.legend(loc=1)
#
#         #ylim = {'GF2':1e-2, 'WV2':5e-4, 'WV3':1e-2}
#         # ylim = {'GF2': 1e-3}
#         # plt.ylim(0, ylim[self.data_type[0:3]])
#         plt.ylim(0, 1e-3)
#         plt.savefig(self.path + 'loss.jpg', dpi=300)
#         plt.clf()


import os
import torch
import matplotlib.pyplot as plt
import numpy as np


class Evaluate():
    def __init__(self, exp_type, data_type, device):
        self.exp_type = exp_type
        self.data_type = data_type
        self.device = device

        # 1. 设定基础路径 down_results/PanNet
        base_dir = os.path.join('down_results', exp_type)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 2. 自动递增文件夹: WV2_0, WV2_1 ...
        i = 0
        while True:
            # 目标目录: down_results/PanNet/WV2_0
            self.save_dir = os.path.join(base_dir, f"{data_type}_{i}")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                break
            i += 1

        print(f"✅ 本次实验结果将保存在: {self.save_dir}")

    def visualize(self, train_loss_list, test_loss_list):
        plt.figure()
        # 绘制训练和测试 Loss
        plt.plot(train_loss_list, label='Train Loss', color='red', linewidth=1.5)
        plt.plot(test_loss_list, label='Test Loss', color='cyan', linestyle='--', linewidth=1.5)

        plt.title(f'Loss Curve: {self.exp_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='-.')

        # 保存图片
        save_path = os.path.join(self.save_dir, 'loss_curve.jpg')
        plt.savefig(save_path, dpi=300)
        plt.close()  # 关闭画布，释放内存

    # 功能2 & 3: 保存最佳模型参数 和 详细指标 Log
    def save_best(self, model, epoch, metrics_dict):
        """
        metrics_dict: 包含所有指标的字典，例如 {'PSNR': 30.1, 'SSIM': 0.8...}
        """
        # 1. 保存模型参数 .pth
        model_save_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_save_path)

        # 2. 保存指标到 log.txt
        log_path = os.path.join(self.save_dir, 'best_metrics_log.txt')
        with open(log_path, 'w') as f:
            f.write(f"Best Result Achieved at Epoch: {epoch}\n")
            f.write("=" * 30 + "\n")
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value:.6f}\n")

        print(f"🏆 Epoch {epoch} 创新高! 模型已保存, 指标已写入日志.")

class EvaluateFull():
    def __init__(self, exp_type, data_type):
        # 1. 设定基础路径 full_results/PanNet
        base_dir = os.path.join('full_results', exp_type)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 2. 文件夹自增逻辑
        i = 0
        while True:
            self.save_dir = os.path.join(base_dir, f"{data_type}_{i}")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                break
            i += 1
        print(f"✅ 全尺度实验结果将保存在: {self.save_dir}")

    def log_metrics(self, metrics_dict):
        log_path = os.path.join(self.save_dir, 'full_scale_metrics_log.txt')
        with open(log_path, 'w') as f:
            f.write("Full Resolution Test Results (Non-reference Metrics)\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value:.6f}\n")
        print(f"📄 指标已记录至: {log_path}")