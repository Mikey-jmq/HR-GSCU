"""Popular image quality assessment methods for pansharpening
Reference: https://github.com/wasaCheney/IQA_pansharpening_python/blob/master/IQAs.py"""

import torch
import torch.nn.functional as F
from utils1 import psnr_loss, ssim, sam, SSIM

eps = torch.finfo(torch.float32).eps

def get_metrics_reduced(img1, img2):
    # input: img1 {the pan-sharpened image}, img2 {the ground-truth image}
    # return: (larger better) psnr, ssim, scc, (smaller better) sam, ergas
    m1 = psnr_loss(img1, img2, 1.)
    m2 = ssim(img1, img2, 11, 'mean', 1.)
    m3 = cc(img1, img2)
    m31 = cc(img1, img2).mean()
    m4 = sam(img1, img2)
    m5 = ergas(img1, img2)
    #return [m1.item(), m2.item(), m3.item(), m4.item(), m5.item()]
    return [m1.item(), m2.item(), m31.item(), m4.item(), m5.item()]

def ergas(img_fake, img_real, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    
    N,C,H,W = img_real.shape
    means_real = img_real.reshape(N,C,-1).mean(dim=-1)
    mses = ((img_fake - img_real)**2).reshape(N,C,-1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS
    
    return 100 / scale * torch.sqrt((mses / (means_real**2 + eps)).mean())
    
def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N,C,_,_ = img1.shape
    img1 = img1.reshape(N,C,-1)
    img2 = img2.reshape(N,C,-1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / ( eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)) )
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)

def qindex(img1, img2):
    # get the q index (universal quality index proposed by Wang and Bovik 2002).
    # q index is the special case of SSIM by setting C1=C2=0.
    # (see page 605 Eq. (13) in Wang et al., IEEE TIP, vol.13, no.4, pp.600-612)
    _qindex = SSIM(window_size=11, reduction="none", max_val=1.)
    # _qindex.C1 = eps
    # _qindex.C2 = eps
    output = 1-2*_qindex(img1, img2)
    return output.reshape(output.shape[0],-1).mean(dim=-1)

def D_lambda(img_fake, img_ms, p=1.):
    N,C,_,_ = img_fake.shape
    q_fake = torch.zeros((N,int(C*(C-1)/2)))
    q_ms   = torch.zeros((N,int(C*(C-1)/2)))
    
    k = 0
    for i in range(C):
        for j in range(i+1,C):
            q_fake[:,k] = qindex(img_fake[:,i:i+1,:,:], img_fake[:,j:j+1,:,:])
            q_ms[:,k]   = qindex(img_ms[:,i:i+1,:,:],   img_ms[:,j:j+1,:,:])
            k = k+1

    return (q_fake-q_ms).abs().pow(p).mean(-1).pow(1/p)

def D_s(img_fake, img_ms, img_pan, q=1.):
    N,C,H,W = img_fake.shape
    _,_,h,w = img_ms.shape
    lr_pan = F.interpolate(img_pan, size=[h,w], mode='bicubic', align_corners=True)
    
    q_hr = torch.zeros((N,C))
    q_lr = torch.zeros((N,C))
    k = 0
    for i in range(C):
        q_hr[:,k] = qindex(img_fake[:,i:i+1,:,:], img_pan)
        q_lr[:,k] = qindex(img_ms[:,i:i+1,:,:], lr_pan)
        k = k+1
        
    return (q_hr-q_lr).abs().pow(q).mean(-1).pow(1/q)

def QNR(img_fake, img_ms, img_pan, a=1., b=1., q=1., p=1.):
    d_lambda = D_lambda(img_fake, img_ms, p)
    d_s = D_s(img_fake, img_ms, img_pan, q)
    qnr = (1-d_lambda).pow(a)*(1-d_s).pow(b)
    return qnr.mean(), d_lambda.mean(), d_s.mean()

    
    
    