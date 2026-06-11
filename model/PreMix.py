from model.PGCU import *
from model.gscu_final import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SFAU import *
from model.network import  *
import kornia

def _high_pass_filter(img, ksize=(3, 3), sigma=(1.5, 1.5)):
    blur = kornia.filters.gaussian_blur2d(img, ksize, sigma)
    high_pass_filtered = img - blur
    return high_pass_filtered


def _equalize_clahe(img):
    return kornia.enhance.equalize_clahe(torch.clip(img, 0, 1))

class PreMixHuge(nn.Module):
    def __init__(self, ms_channels, embed_dim=32, kernel_size=3, pf_kernel=3,enable_EWFM=True, num_layers=3, beta=0.1, act="tanh+relu"):
        super().__init__()

        self.model1 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])
        self.model2 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])
        self.model3 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])

    def upsample(self, x, h, w):
        return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)

    def _forward_model(self, model, ms, pan, filtered_ms, filtered_pan, last_ms_f=None, last_pan_f=None):
        ms2x = F.interpolate(ms, scale_factor=0.5)
        ms4x = F.interpolate(ms, scale_factor=0.25)
        pan2x = F.interpolate(pan, scale_factor=0.5)
        pan4x = F.interpolate(pan, scale_factor=0.25)

        f4x, last_ms_f, last_pan_f = model[2](ms4x, pan4x, filtered_ms, filtered_pan, last_ms_f, last_pan_f)
        f2x, last_ms_f, last_pan_f = model[1](ms2x, pan2x, f4x, filtered_pan, last_ms_f, last_pan_f)
        f, last_ms_f, last_pan_f = model[0](ms, pan, f2x, filtered_pan, last_ms_f, last_pan_f)
        return f, last_ms_f, last_pan_f

    def forward(self, ms, pan):
        ms_up = self.upsample(ms, pan.shape[-2], pan.shape[-1])
        hp_ms_up = _high_pass_filter(ms_up)
        hp_pan = _high_pass_filter(pan)
        eq_ms_up = _equalize_clahe(ms_up)
        eq_pan = _equalize_clahe(pan)
        f_hp_ms, last_ms_f, last_pan_f = self._forward_model(self.model3, hp_ms_up, hp_pan, ms_up, pan)
        f_grad_ms, last_ms_f, last_pan_f = self._forward_model(self.model2, eq_ms_up, eq_pan, f_hp_ms, pan, last_ms_f,last_pan_f)
        f_ms, last_ms_f, last_pan_f = self._forward_model(self.model1, ms_up, pan, f_grad_ms, pan, last_ms_f,last_pan_f)

        pred = f_ms + ms_up
        return pred


class PreMixHuge_PGCU(nn.Module):
    def __init__(self, ms_channels, embed_dim=32, kernel_size=3, pf_kernel=3, enable_EWFM=True, num_layers=3, beta=0.1,
                 act="tanh+relu"):
        super().__init__()

        self.PGCU = PGCU(4, 128)

        self.model1 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])
        self.model2 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])
        self.model3 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])

    def _forward_model(self, model, ms, pan, filtered_ms, filtered_pan, last_ms_f=None, last_pan_f=None):
        ms2x = F.interpolate(ms, scale_factor=0.5)
        ms4x = F.interpolate(ms, scale_factor=0.25)
        pan2x = F.interpolate(pan, scale_factor=0.5)
        pan4x = F.interpolate(pan, scale_factor=0.25)

        f4x, last_ms_f, last_pan_f = model[2](ms4x, pan4x, filtered_ms, filtered_pan, last_ms_f, last_pan_f)
        f2x, last_ms_f, last_pan_f = model[1](ms2x, pan2x, f4x, filtered_pan, last_ms_f, last_pan_f)
        f, last_ms_f, last_pan_f = model[0](ms, pan, f2x, filtered_pan, last_ms_f, last_pan_f)
        return f, last_ms_f, last_pan_f

    def forward(self, ms, pan):
        ms_up = self.PGCU(pan, ms)
        hp_ms_up = _high_pass_filter(ms_up)
        hp_pan = _high_pass_filter(pan)
        eq_ms_up = _equalize_clahe(ms_up)
        eq_pan = _equalize_clahe(pan)
        f_hp_ms, last_ms_f, last_pan_f = self._forward_model(self.model3, hp_ms_up, hp_pan, ms_up, pan)
        f_grad_ms, last_ms_f, last_pan_f = self._forward_model(self.model2, eq_ms_up, eq_pan, f_hp_ms, pan, last_ms_f,
                                                               last_pan_f)
        f_ms, last_ms_f, last_pan_f = self._forward_model(self.model1, ms_up, pan, f_grad_ms, pan, last_ms_f,
                                                          last_pan_f)

        pred = f_ms + ms_up
        return pred

class PreMixHuge_GSCU(nn.Module):
    def __init__(self, ms_channels, embed_dim=32, kernel_size=3, pf_kernel=3,enable_EWFM=True, num_layers=3, beta=0.1, act="tanh+relu"):
        super().__init__()

        self.GSCU = GaussianSplatter(kernel_size=5, c1=6, channels=ms_channels, n_feats=48, n_classes=50)

        self.model1 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])
        self.model2 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])
        self.model3 = nn.ModuleList(
            [PreMixModel(i, ms_channels, embed_dim, kernel_size, pf_kernel, enable_EWFM, num_layers, beta, act) for i in
             range(3)])

    def _forward_model(self, model, ms, pan, filtered_ms, filtered_pan, last_ms_f=None, last_pan_f=None):
        ms2x = F.interpolate(ms, scale_factor=0.5)
        ms4x = F.interpolate(ms, scale_factor=0.25)
        pan2x = F.interpolate(pan, scale_factor=0.5)
        pan4x = F.interpolate(pan, scale_factor=0.25)

        f4x, last_ms_f, last_pan_f = model[2](ms4x, pan4x, filtered_ms, filtered_pan, last_ms_f, last_pan_f)
        f2x, last_ms_f, last_pan_f = model[1](ms2x, pan2x, f4x, filtered_pan, last_ms_f, last_pan_f)
        f, last_ms_f, last_pan_f = model[0](ms, pan, f2x, filtered_pan, last_ms_f, last_pan_f)
        return f, last_ms_f, last_pan_f

    def forward(self, ms, pan):
        ms_up = self.GSCU(pan, ms)
        hp_ms_up = _high_pass_filter(ms_up)
        hp_pan = _high_pass_filter(pan)
        eq_ms_up = _equalize_clahe(ms_up)
        eq_pan = _equalize_clahe(pan)
        f_hp_ms, last_ms_f, last_pan_f = self._forward_model(self.model3, hp_ms_up, hp_pan, ms_up, pan)
        f_grad_ms, last_ms_f, last_pan_f = self._forward_model(self.model2, eq_ms_up, eq_pan, f_hp_ms, pan, last_ms_f,last_pan_f)
        f_ms, last_ms_f, last_pan_f = self._forward_model(self.model1, ms_up, pan, f_grad_ms, pan, last_ms_f,last_pan_f)

        pred = f_ms + ms_up
        return pred
