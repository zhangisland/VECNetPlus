import torch
import torch.nn as nn


class CrossFrameIAN(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # for concat [ ref, others, diff(others-ref) ]
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, ref_f, other_f):
        """
        ref_f: [B, 1, C, H, W]
        other_f: [B, N-1, C, H, W]
        return: att: [B, N-1, C, 1, 1]
        """
        B, F, C, H, W = other_f.shape

        # broadcast
        ref_f = ref_f.expand(-1, F, -1, -1, -1)  # [B, N-1, C, H, W]

        # flatten batch+frame for pooling + conv
        ref_flat = ref_f.reshape(B*F, C, H, W)
        oth_flat = other_f.reshape(B*F, C, H, W)
        diff_flat = (other_f - ref_f).reshape(B*F, C, H, W)

        # global pooling
        ref_g = self.avgpool(ref_flat)
        oth_g = self.avgpool(oth_flat)
        diff_g = self.avgpool(diff_flat)

        feat = torch.cat([ref_g, oth_g, diff_g], dim=1)  # [B*F, 3C,1,1]
        att_flat = self.mlp(feat)                         # [B*F, C,1,1]

        att = att_flat.view(B, F, C, 1, 1)
        return att
