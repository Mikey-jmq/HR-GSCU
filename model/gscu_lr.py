import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

from utils1 import psnr_loss


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])#(H*W,2)坐标对
    rgb = img.view(3, -1).permute(1, 0)#(H*W,3)rgb对,permute函数是改变顺序的，这里表示交换
    return coord, rgb

def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords

def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-0, 0].
    """
    # Normalize pixel coordinates to [-0, 0] range归一化
    input_coords = input_coords.to(image_tensor.device)#设备调整
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0#将坐标从 [0, 0] 转换为 [-0, 0] 范围

    # Fetching the colour of the pixels in each coordinates抓取像素颜色值
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)#改变维度为[batch,N,2]
       #提取 x 和 y 坐标，并转换为整数类型
    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)#生成批次索引

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]#

    return color_values, coords

def extract_patch(image, center, radius, padding_mode='constant'):
    """
    Extract a patch from an image with the specified center and radius.
    Args:
        image (torch.Tensor): Input image of shape [batch_size, channels, height, width].
        center (tuple): Coordinates (y, x) of the patch center.
        radius (int): Radius of the patch.
        padding_mode (str, optional): Padding mode, can be 'constant', 'reflect', 'replicate', or 'circular'. Default is 'constant'.

    Returns:
        torch.Tensor: Extracted patch of shape [batch_size, channels, 2 * radius, 2 * radius].
    """
    height, width = image.shape[-2:]

    # Convert center coordinates to integers
    center_y, center_x = int(round(center[0])), int(round(center[1]))

    # Calculate patch boundaries计算补丁边界
    top = center_y - radius
    bottom = center_y + radius
    left = center_x - radius
    right = center_x + radius

    # Check if boundaries are out of image bounds计算超出图像边界的填充量
    top_padding = max(0, -top)
    bottom_padding = max(0, bottom - height)
    left_padding = max(0, -left)
    right_padding = max(0, right - width)

    # Pad the image填充
    padded_image = torch.nn.functional.pad(image, (left_padding, right_padding, top_padding, bottom_padding),
                                           mode=padding_mode)

    # Extract the patch
    patch = padded_image[..., top_padding:top_padding + 2 * radius, left_padding:left_padding + 2 * radius]

    return patch

class Encoder(nn.Module):
    def __init__(self, channels=4, n_feats=48,n_classes=50):
        """
        channel:ms的通道
        n_feats:特征通道
        t：通道特异性向量V的通道
        """
        super(Encoder, self).__init__()
        self.n_feats = n_feats
        self.pan_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        ####通道特异性V的分支
        self.v_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, n_feats, kernel_size=1)
        )
        #SE通道注意力
        self.v_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_feats // 4, n_feats, kernel_size=1),
            nn.Sigmoid()
        )
        ####特征计算分支
        self.downsample_pan = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.downsample_fusion_mid = nn.Sequential(
            nn.Conv2d(channels * 2, n_feats, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.downsample_fusion_low = nn.Sequential(
            nn.Conv2d(channels * 2, n_feats, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, n_feats // 4, kernel_size=3, padding=ks, dilation=ks),
                nn.ReLU()
            ) for ks in [1, 2, 3]
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*2, n_feats // 4, kernel_size=1),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 2, n_feats, kernel_size=1),
            nn.ReLU()
        )
        self.lateral = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.feat_fusion = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        # Logits（基于 feat+coord）
        self.coord_conv = nn.Conv2d(2, channels*4, kernel_size=1)
        self.logits_branch = nn.Conv2d(n_feats+channels*4, n_classes+1,kernel_size=1)

    def forward(self, ms, pan):
        target_h, target_w = ms.shape[-2:]
        down_pan = self.downsample_pan(pan)
        down_input = torch.cat([ms, down_pan], dim=1)#B 8 32 32

        # V 分支
        V = self.v_fusion(down_input)
        channel_weights = self.v_attention(V)
        V = V * channel_weights#B 48 32 32

        # 跨模态特征
        # down_pan = self.downsample_pan(pan)
        # down_input = torch.cat([ms, down_pan], dim=1)
        up_mid = self.downsample_fusion_mid(down_input)
        low_feat = self.downsample_fusion_low(down_input)
        up_low = F.interpolate(low_feat, size=(target_h, target_w), mode='bilinear')

        feats = [branch(down_input) for branch in self.dilated]
        gp = self.global_pool(down_input)
        gp = gp.expand(-1, -1, target_h, target_w)

        feat = torch.cat(feats, dim=1)
        feat = torch.cat([feat, gp], dim=1)
        residual = self.residual(down_input)
        feat = feat + residual
        feat = self.lateral(feat)
        feat = feat + up_mid + up_low
        feat = self.feat_fusion(feat)

        # Logits
        coords = make_coord(ms.shape[-2:], flatten=False).to(pan.device)
        coords = coords.permute(2, 0, 1).unsqueeze(0).expand(pan.shape[0], -1, -1, -1)
        coord_feat = self.coord_conv(coords)
        logits_input = torch.cat([feat, coord_feat], dim=1)
        logits = self.logits_branch(logits_input)

        B, Class, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(B * H * W, Class)
        logits = F.gumbel_softmax(logits, tau=1, hard=False)
        logits = logits.view(B, H, W, Class).permute(0, 3, 1, 2).contiguous()

        return V, feat, logits

class GaussianSplatter(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, kernel_size=5, unfold_row=8, unfold_column=8, c1=6, channels=4, n_feats=48,n_classes=50):
        """
        Initialize the 2D Gaussian Splatter module.
        Args:
            kernel_size (int): The size of the kernel to convert rasterization.
            unfold_row (int): The number of points in the row dimension of the Gaussian grid.
            unfold_column (int): The number of points in the column dimension of the Gaussian grid.
            c1: 2DGS's feat.
            channels: LRMS channels.
            n_feats: The channels of features from Encoder.
        """
        super(GaussianSplatter, self).__init__()
        self.encoder = Encoder(channels=channels, n_feats=n_feats,n_classes=n_classes)
        self.feat, self.logits = None, None  
        # Key parameter in 2D Gaussian Splatter参数
        self.kernel_size = kernel_size
        self.row = unfold_row
        self.column = unfold_column
        self.c1 = c1
        self.channels = channels
        self.n_feats = n_feats
        self.c2 = n_feats - c1
        # V2 通道注意力（增强 feat2）
        self.v2_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c2, self.c2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 通道调整
        self.fine_adjust = nn.Sequential(
            nn.Conv2d(n_feats, channels, kernel_size=3, padding=1)
        )
        self.residual = nn.Conv2d(n_feats, channels, kernel_size=1)
        # 高斯分布生成组合
        sigma_x_samples = torch.normal(mean=1.2, std=0.6, size=(n_classes,)).clamp(0, 2.4).cuda()
        sigma_y_samples = torch.normal(mean=1.1, std=0.55, size=(n_classes,)).clamp(0, 2.2).cuda()
        rho_samples = torch.normal(mean=0.3, std=0.6, size=(n_classes,)).clamp(-0.9, 1.5).cuda()
        gau_dict = torch.stack([sigma_x_samples, rho_samples, sigma_y_samples], dim=1)
        gau_dict = torch.cat((gau_dict, torch.zeros(1, 3).cuda()), dim=0)  # 51 组合
        self.sigma_x = gau_dict[:, 0]  # (51,)
        self.rhoxy = gau_dict[:, 1]  # (51,)
        self.sigma_y = gau_dict[:, 2]  # (51,)

    def weighted_gaussian_parameters(self, logits):
        """
        Computes weighted Gaussian parameters based on logits and the Gaussian kernel parameters (sigma_x, sigma_y, opacity).
        The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial
        location across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension.
        Args:
            logits (torch.Tensor): Logits tensor of shape [batch, class, height, width].
        Returns:
            tuple: A tuple containing the weighted Gaussian parameters:
                - weighted_sigma_x (torch.Tensor): Tensor of shape [height * width] representing the weighted x-axis standard deviations.
                - weighted_sigma_y (torch.Tensor): Tensor of shape [height * width] representing the weighted y-axis standard deviations.
        Description:
            This function computes weighted Gaussian parameters based on the input tensor, logits, and the provided Gaussian kernel parameters (sigma_x, sigma_y, and opacity). The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial location (height and width) across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension, yielding tensors of shape [height * width] for the weighted sigma_x, sigma_y, and opacity.
        """
        batch_size, num_classes, height, width = logits.size()
        logits = logits.permute(0, 2, 3, 1)  # Reshape logits to [batch, height, width, class]
        device = logits.device
        sigma_x = self.sigma_x.to(device)
        sigma_y = self.sigma_y.to(device)
        rhoxy = self.rhoxy.to(device)

        # Compute weighted sum of Gaussian parameters across class dimension（按类别体现在sum(dim=-0)
        weighted_sigma_x = (logits * sigma_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_sigma_y = (logits * sigma_y.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_rhoxy = (logits * rhoxy.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # Reshape and average across batch dimension
        weighted_sigma_x = weighted_sigma_x.reshape(batch_size, -1).mean(dim=0)
        weighted_sigma_y = weighted_sigma_y.reshape(batch_size, -1).mean(dim=0)
        weighted_rhoxy = weighted_rhoxy.reshape(batch_size, -1).mean(dim=0)

        return weighted_sigma_x, weighted_sigma_y, weighted_rhoxy

    def gen_feat(self, ms, pan):
        """Generate feature and logits by encoder."""
        self.ms = ms
        self.pan = pan
        self.scale = pan.shape[-1]/ms.shape[-1]
        self.V, self.feat, self.logits = self.encoder.forward(ms, pan)
        return self.feat, self.logits

    def query_rgb(self):
        # 1. Get LR feature and logits
        B, _, curr_H, curr_W = self.feat.shape#B 48 32 32
        feat, lr_feat, logits = self.feat[:, :self.c1, :, :], self.feat[:, self.c1:, :,:], self.logits  # Channel decoupling
        # feat(B,c1,32,32) lr_feat(B,n_feats-c1,32,32) #logits(B,n_classed,32,32)
        feat_size, feat_device = feat.shape, feat.device
        V1, V2 = self.V[:, :self.c1, :, :], self.V[:, self.c1:, :, :]

        # 3. Unfold the feature / logits to many small patches to avoid extreme GPU memory consumption
        num_kernels_row = math.ceil(feat_size[-2] / self.row)
        num_kernels_column = math.ceil(feat_size[-1] / self.column)
        upsampled_size = (num_kernels_row * self.row, num_kernels_column * self.column)
        upsampled_inp = F.interpolate(feat, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_logits = F.interpolate(logits, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_V1 = F.interpolate(V1, size=upsampled_size, mode='bicubic', align_corners=False)
        unfold = nn.Unfold(kernel_size=(self.row, self.column), stride=(self.row, self.column))
        unfolded_feature = unfold(upsampled_inp).contiguous()
        unfolded_logits = unfold(upsampled_logits).contiguous()
        unfolded_V1 = unfold(upsampled_V1).contiguous()
        # Unfolded_feature dimension becomes [Batch, C*K*K, L], where L is the number of columns after unfolding
        L = unfolded_feature.shape[-1]
        unfold_feat = unfolded_feature.transpose(1, 2).contiguous().reshape(feat_size[0] * L, feat_size[1], self.row,self.column)#B 6 8 8
        unfold_logits = unfolded_logits.transpose(1, 2).contiguous().reshape(logits.shape[0] * L, logits.shape[1],self.row, self.column)#B 51 8 8
        unfold_V1 = unfolded_V1.transpose(1, 2).contiguous().reshape(V1.shape[0] * L, V1.shape[1], self.row,self.column)#B 6 8 8

        # 4. Generate colors_(features) and coords_norm
        coords_ = generate_meshgrid(unfold_feat.shape[-2], unfold_feat.shape[-1])
        num_LR_points = unfold_feat.shape[-2] * unfold_feat.shape[-1]
        colors, coords_norm = fetching_features_from_tensor(unfold_feat, coords_)

        # 5. Rasterization: Generating grid
        # 5.0. Spread Gaussian points over the whole feature map
        batch_size, channel, _, _ = unfold_feat.shape
        weighted_sigma_x, weighted_sigma_y, weighted_rhoxy = self.weighted_gaussian_parameters(unfold_logits)
        sigma_x = weighted_sigma_x.view(num_LR_points, 1, 1)
        sigma_y = weighted_sigma_y.view(num_LR_points, 1, 1)
        rhoxy = weighted_rhoxy.view(num_LR_points, 1, 1)

        # 5.2. Gaussian expression
        covariance = torch.stack(
            [torch.stack([sigma_x ** 2 + 1e-5, rhoxy], dim=-1),
             torch.stack([rhoxy, sigma_y ** 2 + 1e-5], dim=-1)], dim=-2
        )  # when correlation rou is set to zero, covariance will always be positive semi-definite
        inv_covariance = torch.inverse(covariance).to(feat_device)#64 1 1 2 2

        # 5.3. Choosing a broad range for the distribution [-5,5] to avoid any clipping
        start = torch.tensor([-5.0], device=feat_device).view(-1, 1)
        end = torch.tensor([5.0], device=feat_device).view(-1, 1)
        base_linspace = torch.linspace(0, 1, steps=self.kernel_size, device=feat_device)
        ax_batch = start + (end - start) * base_linspace
        # Expanding dims for broadcasting
        ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, self.kernel_size, -1)

        # 5.4. Creating a batch-wise meshgrid using broadcasting
        xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
        xy = torch.stack([xx, yy], dim=-1)
        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
        kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=feat_device) *
                                 torch.sqrt(torch.det(covariance)).to(feat_device).view(num_LR_points, 1, 1))
        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
        kernel_normalized = kernel / kernel_max_2  # (num_LR_points,kernel_size,kernel_size)
        unfold_V1_reshaped = unfold_V1.view(batch_size, channel, num_LR_points)  # (B, 8, 64)
        unfold_V1_weights = torch.softmax(unfold_V1_reshaped, dim=1)  # (B, 8, 64)
        kernel_channel = kernel_normalized.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, channel, 1,1)  # (B, 64, 8, kernel_size, kernel_size)

        kernel_color = kernel_channel * unfold_V1_weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 8, kernel_size, kernel_size)
        kernel_color = kernel_color.view(batch_size * num_LR_points, channel, self.kernel_size,self.kernel_size)  # (B*64, 8, kernel_size, kernel_size)

        # 5.5. Adding padding to make kernel size equal to the image size
        pad_h = round(unfold_feat.shape[-2]*self.scale) - self.kernel_size
        pad_w = round(unfold_feat.shape[-1]*self.scale) - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller or equal to the image size.")
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        kernel_color_padded = torch.nn.functional.pad(kernel_color, padding, "constant", 0)

        # 5.6. Create a batch of 2D affine matrices
        _, c, h, w = kernel_color_padded.shape  # num_LR_points*batch_size, channel, hr_h, hr_w
        theta = torch.zeros(batch_size, num_LR_points, 2, 3, dtype=torch.float32, device=feat_device)
        theta[:, :, 0, 0] = 1.0
        theta[:, :, 1, 1] = 1.0
        theta[:, :, :, 2] = coords_norm
        grid = F.affine_grid(theta.view(-1, 2, 3), size=[batch_size * num_LR_points, c, h, w],align_corners=True).contiguous()
        kernel_color_padded_translated = F.grid_sample(kernel_color_padded.contiguous(), grid.contiguous(),align_corners=True)
        kernel_color_padded_translated = kernel_color_padded_translated.view(batch_size, num_LR_points, c, h, w)

        # 6. Apply Gaussian splatting
        # colors_.shape = [batch, num_LR_points, channel], colors.shape = [batch, num_LR_points, channel]
        color_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1)
        final_image_layers = color_values_reshaped * kernel_color_padded_translated
        final_image = final_image_layers.sum(dim=1)
        final_image = torch.clamp(final_image, 0, 1)

        # 7. Fold the input back to the original size
        # Calculate the number of kernels needed to cover each dimension.
        kernel_h, kernel_w = round(self.row*self.scale), round(self.column*self.scale)
        fold = nn.Fold(output_size=(kernel_h * num_kernels_row, kernel_w * num_kernels_column),
                       kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        final_image = final_image.reshape(feat_size[0], L, feat_size[1] * kernel_h * kernel_w).transpose(1, 2)
        final_image = fold(final_image)
        final_image = F.interpolate(final_image, size=(curr_H*4, curr_W*4), mode='bicubic', align_corners=False)

        # 8. Decoder and bicubic
        # V2 注意力增强 feat2
        v2_weights = self.v2_attention(V2)
        lr_feat = F.interpolate(lr_feat, size=(curr_H*4, curr_W*4), mode='bicubic', align_corners=False)
        lr_feat = lr_feat * v2_weights  # (B,c2,128,128)
        # 拼接final_image 和 lr_feat
        feat = torch.cat([final_image, lr_feat], dim=1)  # (B, n_feats, 128, 128)
        out = self.fine_adjust(feat)  # (B, 4, 128, 128)
        out = out + self.residual(feat)  # (B, 4, 128, 128)
        return out

    def forward(self, pan, lrms):
        self.gen_feat(lrms, pan)
        return self.query_rgb()


if __name__ == '__main__':
    # A simple example of implementing class GaussianSplatter
    model = GaussianSplatter()
    lrms = torch.rand(1, 4, 32, 32)
    pan = torch.rand(1, 1, 128, 128)
    pred = model.forward(pan,lrms)
    print(pred.shape)
    # Encoder = Encoder(4,48,100)
    # V,feat,Logits = Encoder.forward(lrms,pan)
    # print(V.shape,feat.shape,Logits.shape)
