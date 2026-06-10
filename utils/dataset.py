import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio
from utils.funcation import *
    
    
# preprocess data in different dataset
def Preprocess(pan, ms, up_type):
    lrms = ms.resize((int(ms.size[0]/4), int(ms.size[1]/4)), Image.BICUBIC)  # 下采样得到lrms图像
    up_ms = upsampling(lrms, ms.size, up_type)
    ms = np.array(ms).transpose(2, 0, 1) / 255
    lrms = np.array(lrms).transpose(2, 0, 1) / 255
    up_ms = np.array(up_ms).transpose(2, 0, 1) / 255
    pan = np.expand_dims(np.array(pan), axis=0) / 255
    return ms, pan, lrms, up_ms, highpass(pan), highpass(lrms)


def Preprocess1(pan, ms, up_type):
    # 先把原始的 PIL 图像转换为标准的 numpy 数组 (H, W, C) 和 (H, W)
    ms_np = np.array(ms) / 255.0
    pan_np = np.array(pan) / 255.0

    # 统一进行转置，将 (H, W, C) 变为标准的 (C, H, W)
    # 因为在全尺度测试中，输入多光谱 lrms 就等于真实的 ms
    ms_tensor = ms_np.transpose(2, 0, 1)
    lrms_tensor = ms_np.transpose(2, 0, 1)

    # 全色图增加通道维度，变为 (1, H, W)
    pan_tensor = np.expand_dims(pan_np, axis=0)

    # 注意：如果 highpass 函数是你自定义的，确保它接受并返回 (C, H, W) 格式的数据
    return ms_tensor, pan_tensor, lrms_tensor, highpass(pan_tensor), highpass(lrms_tensor)

# dataset
class MyDataset(Dataset):
    
    def __init__(self, root, ms_path, pan_path, up_type):
        super(MyDataset, self).__init__()
        self.root = root
        self.ms_path = ms_path
        self.pan_path = pan_path
        self.ms_list = os.listdir(root+'/'+ms_path)
        self.pan_list = os.listdir(root+'/'+pan_path)
        self.up_type = up_type
        
    def __getitem__(self, index):
        ms = Image.open(self.root+'/'+self.ms_path+'/'+self.ms_list[index])
        pan = Image.open(self.root+'/'+self.pan_path+'/'+self.pan_list[index])
            
        ms, pan, lrms, up_ms, hpan, hlrms = Preprocess(pan, ms, self.up_type)
        return ms, pan, lrms, up_ms, hpan, hlrms
    
    def __len__(self):
        return len(self.ms_list)


class MyDataset1(Dataset):

    def __init__(self, root, ms_path, pan_path, up_type):
        super(MyDataset1, self).__init__()
        self.root = root
        self.ms_path = ms_path
        self.pan_path = pan_path
        self.ms_list = os.listdir(root + '/' + ms_path)
        self.pan_list = os.listdir(root + '/' + pan_path)
        self.up_type = up_type

    def __getitem__(self, index):
        ms = Image.open(self.root + '/' + self.ms_path + '/' + self.ms_list[index])
        pan = Image.open(self.root + '/' + self.pan_path + '/' + self.pan_list[index])
        ms, pan, lrms, hpan, hlrms = Preprocess1(pan, ms, self.up_type)
        return ms, pan, lrms, hpan, hlrms

    def __len__(self):
        return len(self.ms_list)


def Preprocess_Full(pan, ms):
    ms_np = np.array(ms).transpose(2, 0, 1) / 255.0
    pan_np = np.expand_dims(np.array(pan), axis=0) / 255.0

    hpan = highpass(pan_np)
    hlrms = highpass(ms_np)

    return pan_np, ms_np, hpan, hlrms


class MyDatasetFull(Dataset):
    def __init__(self, root, ms_path, pan_path):
        super(MyDatasetFull, self).__init__()
        self.root = root
        self.ms_list = sorted(os.listdir(os.path.join(root, ms_path)))
        self.pan_list = sorted(os.listdir(os.path.join(root, pan_path)))
        self.ms_path = ms_path
        self.pan_path = pan_path

    def __getitem__(self, index):
        ms = Image.open(os.path.join(self.root, self.ms_path, self.ms_list[index]))
        pan = Image.open(os.path.join(self.root, self.pan_path, self.pan_list[index]))

        pan, lrms, hpan, hlrms = Preprocess_Full(pan, ms)
        return pan, lrms, hpan, hlrms

    def __len__(self):
        return len(self.ms_list)


class MyDatasetFull2(Dataset):

    def __init__(self, root, ms_path, pan_path, up_type):
        super(MyDatasetFull2, self).__init__()
        self.root = root
        self.ms_path = ms_path
        self.pan_path = pan_path
        self.ms_list = os.listdir(root + '/' + ms_path)
        self.pan_list = os.listdir(root + '/' + pan_path)
        self.up_type = up_type

    def __getitem__(self, index):
        ms = Image.open(self.root + '/' + self.ms_path + '/' + self.ms_list[index])
        pan = Image.open(self.root + '/' + self.pan_path + '/' + self.pan_list[index])

        ms, pan, lrms, up_ms, hpan, hlrms = Preprocess(pan, ms, self.up_type)
        return pan, lrms, hpan, hlrms

    def __len__(self):
        return len(self.ms_list)

def Preprocess_Full_Mat(pan, ms):
    ms = ms.astype(np.float32) / 2047.0
    pan = pan.astype(np.float32) / 2047.0
    hpan_raw = highpass(pan)
    hlrms_raw = highpass(ms)
    pan_np = np.expand_dims(pan, axis=0)  # (1, 1024, 1024)
    ms_np = ms.transpose(2, 0, 1)  # (4, 256, 256)

    hpan = np.expand_dims(hpan_raw, axis=0)  # (1, 1024, 1024)
    hlrms = hlrms_raw.transpose(2, 0, 1)  # (4, 256, 256)

    return pan_np, ms_np, hpan, hlrms


class MyDatasetFull_Mat(Dataset):
    def __init__(self, root):
        super(MyDatasetFull_Mat, self).__init__()
        self.root = root
        # 获取目录下所有的 .mat 文件并排序，确保顺序固定
        self.file_list = sorted([f for f in os.listdir(root) if f.endswith('.mat')])

        if len(self.file_list) == 0:
            print(f"⚠️ 警告：在路径 {root} 下没有找到 .mat 文件！")

    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.file_list[index])
        data = sio.loadmat(file_path)
        ms = data['I_MS']
        pan = data['I_PAN']
        pan_np, lrms_np, hpan_np, hlrms_np = Preprocess_Full_Mat(pan, ms)
        return pan_np, lrms_np, hpan_np, hlrms_np

    def __len__(self):
        return len(self.file_list)