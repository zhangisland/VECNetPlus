from loguru import logger
import os
import os.path as osp
import math
import torch
from torch.nn.modules.loss import _Loss
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
import torch.nn.functional as F
import lpips
from torch.autograd import Variable
import torch.nn as nn
from math import exp
from loguru import logger
from torchvision.models import vgg16
import kornia
import numpy as np
from pytorch_msssim import ssim, SSIM
from losses.loss_registry import register_loss

@register_loss('ColorConstancyZeroDCELoss')
class ColorConstancyZeroDCELoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x = batch['pred']

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k.mean()


@register_loss('PSNRLoss')
class PSNRLoss(_Loss):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super().__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, batch):
        pred, target = batch["pred"], batch["gt"]  # 0-1 data range
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

@register_loss('SobelLoss')
class SobelLoss(_Loss):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: none, mean, sum')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.spatial_gradient = kornia.filters.SpatialGradient()

    def forward(self, batch, weight=None, **kwargs):
        output, gt_img = batch["pred"], batch["gt"]
        output_grad = self.spatial_gradient(output)
        gt_grad = self.spatial_gradient(gt_img)

        # Flatten the gradient tensors to compute the L1 loss
        output_grad_flat = torch.cat([output_grad[:, 0], output_grad[:, 1]], dim=1)
        gt_grad_flat = torch.cat([gt_grad[:, 0], gt_grad[:, 1]], dim=1)

        # Compute the L1 loss between the gradients
        loss = torch.abs(output_grad_flat - gt_grad_flat)

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return self.loss_weight * torch.mean(loss)
        elif self.reduction == 'sum':
            return self.loss_weight * torch.sum(loss)
        else:  # self.reduction == 'none'
            return self.loss_weight * loss

@register_loss('L1CharbonnierLoss')
class L1CharbonnierLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.scale = 1.0

    def forward(self, batch):
        x, y = batch["pred"], batch["gt"]
        diff = torch.add(x, -y)  # pred-gt
        diff_sq = diff * diff  # (pred-gt)**2
        diff_sq = torch.mean(diff_sq, 1, True) 
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)
        return loss*self.scale

@register_loss('CB_SSIM_Loss')
class CB_SSIM_Loss(_Loss):
    def __init__(self, alpha=0.8, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, batch, epoch=None):
        x = batch['pred']
        y = batch['gt']
        diff = x - y
        l1_loss = torch.sum(torch.sqrt((diff * diff) + self.eps))

        c = 1.
        for dim in x.shape:
            c *= dim
        
        ssim_module = SSIM(data_range=1., size_average=True, channel=3)
        ssim_loss = (1 - ssim_module(x, y))
        loss = (1 - self.alpha) * l1_loss + self.alpha * c * ssim_loss
        return loss

@register_loss('PerceptualLoss')
class PerceptualLoss(_Loss):
    def __init__(self, backbone='alex'):
        super().__init__()
        device = torch.device("cuda")
        self.backbone = backbone
        # alex net by default
        if backbone == 'alex':
            pretrained_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'ckpts', 'alexnet-owt-4df8aa71.pth')
        elif backbone == 'vgg':
            pretrained_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'ckpts', 'vgg16-397923af.pth')

        if osp.exists(pretrained_path):
            logger.info(f"Loading pretrained model from {pretrained_path}")
            self.lpips_loss = lpips.LPIPS(net=self.backbone, pretrained=True, model_path=pretrained_path).to(device)
        else:
            self.lpips_loss = lpips.LPIPS(net=self.backbone, pretrained=True).to(device)
        

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        loss = self.lpips_loss(pred, gt, normalize=True).mean()
        return loss

@register_loss('VGGPerceptualLoss')
class VGGPerceptualLoss(_Loss):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda")
        self.vgg = vgg16(weights='IMAGENET1K_FEATURES').to(device)
        self.mse_loss = nn.MSELoss()     

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        pred_features = self.vgg(pred)
        gt_features = self.vgg(gt)
        loss = 0.
        for i in range(len(pred_features)):
            loss += self.mse_loss(pred_features[i].detach(), gt_features[i].detach())
        return loss.mean()

# only available if the exposure condition is normal
@register_loss('ColorConstancyLoss')
class ColorConstancyLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def chromaticity_loss(self, pred, target):
        # Convert RGB images to XYZ color space (assuming sRGB)
        pred_xyz = self.rgb_to_xyz(pred)
        target_xyz = self.rgb_to_xyz(target)
        
        # Normalize by dividing by the luminance (Z channel here, for simplicity)
        pred_chromaticity = pred_xyz / pred_xyz[:, :, :, 2:3]  # Normalize by Z channel
        target_chromaticity = target_xyz / target_xyz[:, :, :, 2:3]
        
        # Compute the chromaticity loss (using L2 distance)
        return self.mse_loss(pred_chromaticity, target_chromaticity)

    def rgb_to_xyz(self, rgb):    
        # Reshape to (batch_size * height * width, 3) to apply transformation matrix
        batch_size, channels, height, width = rgb.shape
        rgb_reshaped = rgb.permute(0, 2, 3, 1).reshape(-1, 3)  # Shape: (batch_size * height * width, 3)
        
        # Apply the linear transformation matrix (sRGB to XYZ conversion)
        transform_matrix = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32).to(rgb.device)
        
        # Matrix multiplication: (batch_size * height * width, 3) @ (3, 3) -> (batch_size * height * width, 3)
        xyz_reshaped = torch.matmul(rgb_reshaped, transform_matrix.T)
        
        # Reshape back to (batch_size, height, width, 3)
        return xyz_reshaped.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        # logger.debug(f"pred shape: {pred.shape}, gt shape: {gt.shape}")
        loss = self.chromaticity_loss(pred, gt)

        return loss
        
@register_loss('ColorConstancyFastLoss')
class ColorConstancyFastLoss(_Loss):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eps = eps

        # RGB -> XYZ conversion
        M = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]).float()
        self.rgb2xyz = nn.Conv2d(3, 3, kernel_size=1, bias=False).to(self.device)
        self.rgb2xyz.weight.data[...] = M.view(3, 3, 1, 1)
        self.rgb2xyz.requires_grad_(False)

    def forward(self, preds, targets):
        preds = batch['pred']
        targets = batch['gt']
        # Convert to XYZ
        pred_xyz = self.rgb2xyz(preds)
        gt_xyz   = self.rgb2xyz(targets)

        # Compute log-chroma: log(X) - log(Z), log(Y) - log(Z)
        # log(xyz)-log(z), Gijsenij et al., Color Constancy for Multiple Light Sources
        pred_log = torch.log(pred_xyz[:, :2, :, :] + self.eps) - torch.log(pred_xyz[:, 2:3, :, :] + self.eps)
        gt_log   = torch.log(gt_xyz[:, :2, :, :] + self.eps) - torch.log(gt_xyz[:, 2:3, :, :] + self.eps)

        return self.mse(pred_log, gt_log)

@register_loss('SNRLoss')
class SNRLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.eps = 1e-4
        self.scale = 1.0

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        signal_power = torch.sum(gt ** 2)
        noise_power = torch.sum((gt - pred) ** 2)
        snr = signal_power / (noise_power + 1e-6)
        snr = snr.mean() / 255.0
        
        return -snr


class PSNRMetric(nn.Module):
    # supposed to be 0-1
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(1e-10)

    def forward(self, x, y):
        d = x - y
        mse = torch.mean(d * d) + self.eps
        # consider the ouputs is somewhat out of boundary of [0-1]
        data_range = 1.0 if (x.max() <= 1.0 or y.max() <= 1.0) else 255.0
        psnr = -10 * torch.log10(mse) + 20 * math.log10(data_range)
        return psnr



def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMBasic(nn.Module):
    def __init__(
        self,
        value_range=1.0,
        window_size=11,
        size_average=True,
    ):
        super().__init__()
        self.value_range = value_range
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(self.window_size, self.channel)
        self.eps = 0.00001

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIMMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        ssim_val = ssim(x, y, data_range=1.0, size_average=True)
        return ssim_val
    

@register_loss('FFT_Loss')
class FFT_Loss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, gt = batch["pred"], batch["gt"]
        x = x + 1e-8
        gt = gt + 1e-8
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_amp = torch.abs(x_freq)
        x_phase = torch.angle(x_freq)

        gt_freq = torch.fft.rfft2(gt, norm='backward')
        gt_amp = torch.abs(gt_freq)
        gt_phase = torch.angle(gt_freq)

        loss_amp = torch.mean(torch.sum((x_amp - gt_amp) ** 2))
        loss_phase = torch.mean(torch.sum((x_phase - gt_phase) ** 2))
        return loss_amp, loss_phase


@register_loss('AMP_Loss')
class AMP_Loss(_Loss):
    def __init__(self):
        super().__init__()
        self.cri = nn.L1Loss()

    def forward(self, batch):
        x, y = batch["pred"], batch["gt"]
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag, y_mag).mean()

@register_loss('TV_Loss')
class TV_Loss(_Loss):
    def __init__(self):
        super().__init__()
        self.tv_loss_fun = TV_Single_Loss()

    def forward(self, batch):
        ret_image_illu = self.tv_loss_fun(batch['image_illu'])
        ret_image_illu_v = self.tv_loss_fun(batch['image_illu_v'])
        return ret_image_illu + ret_image_illu_v

# @register_loss('TV_Single_Loss')
class TV_Single_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):        
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


####### L1 loss
@register_loss('L1_Loss')
class L1_Loss(_Loss):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda")
        self.l1loss = nn.L1Loss().to(device)

    def forward(self, batch):
        pred, gt = batch['pred'], batch['gt']
        loss = self.l1loss(pred, gt)
        return loss

####### COLOR loss between (pred, gt)
@register_loss('COLOR_Loss')
class COLOR_Loss(_Loss):
    def __init__(self):
        super().__init__()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)

    def forward(self, batch):
        pred, gt = batch['pred'], batch['gt']
        loss = (1 - self.cos(pred, gt).mean()) * 0.5
        return loss

@register_loss('MASK_Loss')
class MASK_Loss(_Loss):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda")
        self.cri_mask = nn.BCELoss().to(device)

    def forward(self, batch):
        mask = batch['mask']
        thres = batch['thres']
        loss = self.cri_mask(mask, thres)

@register_loss('EXC_Loss')
class EXC_Loss(_Loss):
    # contrastive color correction loss
    def __init__(self, k=5, patch_res=50):
        super().__init__()
        self.D = nn.L1Loss()
        self.k = k  # number of negatives
        self.patch_res = patch_res  # path resolution
        vgg16_model = vgg16(pretrained=True)
        self.color_extractor = vgg16_model.features[:23]

    def Gram(self, mat1, mat2):
        '''
        caculates the Gram matrix

        mat1 (N, 512, 32, 32):              feature map
        mat2 (N, 512, 32, 32):              feature map

        out (N, 512, 512):                  Gram matrix of both feature maps
        '''

        out = []
        for f1, f2 in zip(mat1, mat2):
            out.append(torch.matmul(f1.view(512, -1).T, f2.view(512, -1)))

        return torch.stack(out)

    def forward(self, batch):
        '''
        composition (N, 3, H, W):           composition image (input image to the network)
        pred        (N, 3, H, W):           predicted image (output of the network)
        gt_image    (N, 3, H, W):           ground truth image
        mask        (N, 1, H, W):           image mask
        '''
        pred = batch['pred']
        gt_image = batch['gt']
        lq_image = batch['lq']
        mask = batch['mask']
        B, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        if self.k > pred.size(0) - 1:
            self.k = pred.size(0) - 1

        # calculate foreground and background style representations
        self.SR_extractor = self.color_extractor.to(pred.get_device())
        self.SR_extractor.eval()

        pooled_pred = F.max_pool2d(pred, (4, 4))
        pooled_gt_image = F.max_pool2d(gt_image, (4, 4))
        pooled_lq_image = F.max_pool2d(lq_image, (4, 4))
        pooled_mask = F.max_pool2d(mask, (4, 4))

        f_f = self.SR_extractor(pooled_pred * pooled_mask)
        f_b = self.SR_extractor(pooled_pred * (1 - pooled_mask))
        g_f_plus = self.SR_extractor(pooled_gt_image * pooled_mask)
        g_b_plus = self.SR_extractor(pooled_gt_image * (1 - pooled_mask))
        l_f_minus = self.SR_extractor(pooled_lq_image * pooled_mask)
        l_b_minus = self.SR_extractor(pooled_lq_image * (1 - pooled_mask))

        l_ss_cr = self.D(f_f, g_f_plus) / (self.D(f_f, g_f_plus) + self.D(l_f_minus, g_f_plus) + 1e-8) + \
                  self.D(f_b, g_b_plus) / (self.D(f_b, g_b_plus) + self.D(l_b_minus, g_b_plus) + 1e-8)

        # calculate Gram matrices based on style representations
        c = self.Gram(f_f, f_b)
        c_plus = self.Gram(g_f_plus, g_b_plus)
        c_minus = self.Gram(l_f_minus, l_b_minus)
        l_cs_cr = self.D(c, c_plus) / (self.D(c, c_plus) + self.D(c, c_minus) + 1e-8)

        return l_ss_cr + l_cs_cr
