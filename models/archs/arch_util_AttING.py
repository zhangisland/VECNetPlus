import torch
import torch.nn as nn
from models.archs.arch_util_IAN import CrossFrameIAN


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class AttING_IAN(nn.Module):
    def __init__(self, in_channels, channels, img_channels=3, nframes=5):
        super().__init__()
        # in_channels: 15
        # channels: 16
        self.nframes = nframes
        self.channels = channels

        ## IAN part
        self.instance = nn.InstanceNorm2d(img_channels, affine=True)  # IN (de-exposure) for all frames
        self.map_to_feat_ref = nn.Conv2d(img_channels, channels, 1, 1, 0, bias=False)
        self.map_to_feat_others = nn.Conv2d(img_channels, channels, 1, 1, 0, bias=False)
        self.ian = CrossFrameIAN(channels)
        ## IAN part ends
        
        self.conv1 = nn.Conv2d(nframes * channels, channels, 1, 1, 0, bias=False)
        self.conv2_1 = nn.Conv2d(nframes * channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        self.interative = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels

        self.process = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels * 2, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(2 * channels, channels, 1, 1, 0)


    def forward(self, x):
        # x: [B, N*C, H, W]  e.g. [2, 15, 512, 960]
        B, CC, H, W = x.shape
        N = self.nframes
        C = CC // N

        # 1. IN, de-exposure for all frames
        x_frames = x.view(B*N, C, H, W)
        x_instance_all = self.instance(x_frames)  # [B*N,3,H,W]

        # 2. split frames into `x_ref` and `x_others`
        x_frames = x_instance_all.view(B, N, C, H, W)
        x_ref = x_frames[:, N // 2]  # [B,3,H,W]
        x_others = torch.cat([x_frames[:, :N//2],
                              x_frames[:, N//2+1:]], dim=1)
        x_others = x_others.reshape(B*(N-1), C, H, W)        # [B*(N-1),3,H,W]

        # Channel expansion for sufficient feature representation
        ref_f = self.map_to_feat_ref(x_ref)                # [B,16,H,W]
        others_f = self.map_to_feat_others(x_others)          # [B*4,16,H,W]
        
        # IAN: cross-frame inverted attention
        ref_repeat = ref_f.unsqueeze(1)
        # others_f reshape into [B,4,C,H,W]
        others_f = others_f.view(B, N-1, self.channels, H, W)

        att = self.ian(ref_repeat, others_f)
        # att: [B,4,16,1,1]

        others_f = others_f * att
        others_f = others_f.view(B*(N-1), self.channels, H, W)

        # fusion back to N frames
        fused = torch.cat([
            others_f[:B*(N//2)].view(B, N//2, self.channels, H, W),
            ref_f.unsqueeze(1),
            others_f[B*(N//2):].view(B, N-N//2-1, self.channels, H, W)
        ], dim=1).view(B, N*self.channels, H, W)

        # orginal AttING
        out_identity = self.conv1(fused)                # [B,5*16,H,W] -> [B,16,H,W]
        
        out1 = self.conv2_1(fused)                      # [B,16,H,W]
        out2 = self.conv2_2(out_identity)               # [B,16,H,W]

        out = torch.cat([out1, out2], dim=1)            # [B,32,H,W]

        xp1 = self.interative(out) * out2 + out1
        xp2 = (1 - self.interative(out)) * out1 + out2
        xp = torch.cat([xp1, xp2], 1)

        xp = self.process(self.contrast(xp) + self.avgpool(xp)) * xp
        xp = self.conv1x1(xp)

        return xp, x_instance_all.view(B,CC,H,W)
