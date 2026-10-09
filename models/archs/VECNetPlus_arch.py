import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LAN.NAFNet import ConvBlock, BaselineBlock, framesBlock
from utils.util import load_checkpoint
from models.archs.arch_util import NONLocalBlock2D, ApplyCoeffs, FourAL
from models.archs.arch_util_AttING import AttING_IAN  # add IAN Block in it
from loguru import logger
from torchvision.models.optical_flow import raft_small as RAFT


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle=64, n_fea_in=4, n_fea_out=32, n_fea_map=3):
        super().__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv_map = nn.Conv2d(n_fea_middle, n_fea_map, kernel_size=1, bias=True)  # n,3,h,w

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=3, padding=1, stride=2, bias=False)  # n,c,h//2,w//2
        self.conv3 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=3, padding=1, stride=2, bias=False)  # n,c,h//4,w//4

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)  
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv_map(illu_fea)
        
        illu_fea = self.conv2(illu_fea)
        illu_fea = self.conv3(illu_fea)
        
        return illu_fea, illu_map

class BaselineArch(nn.Module):
    def __init__(self, down_blocks, 
                       latent_dim=32,
                       nframes=1,
                       img_channel=3, 
                       width=16, 
                       middle_blk_num=1, 
                       enc_blk_nums=[], 
                       dec_blk_nums=[], 
                       dw_expand=1,
                       ffn_expand=2,
                       **kwargs,):
        super().__init__()
        
        self.illumination_module = Illumination_Estimator(n_fea_middle=64, n_fea_in=4, n_fea_out=32, n_fea_map=width)

        self.nframes = nframes
        self.center = self.nframes // 2  

        self.frames_Projection = nn.ModuleList(
            [framesBlock(channels=[32, 16, 16, 3], block=ConvBlock) for _ in range(nframes)])

        self.fal = nn.ModuleList([FourAL(img_channel, img_channel) for _ in range(nframes)])

        self.amp_map = nn.Conv2d(in_channels=nframes * img_channel, out_channels=img_channel, kernel_size=1)
        self.pha_maps = nn.ModuleList(
            [nn.Conv2d(in_channels=img_channel, out_channels=img_channel, kernel_size=1) for _ in range(nframes)])

        
        self.attin = AttING_IAN(in_channels=nframes * img_channel, channels=width, nframes=self.nframes)
        
        
        self.intro = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.end_illu = nn.Conv2d(in_channels=width, out_channels=img_channel * 4, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.up_illus = nn.ModuleList()
        self.de_illus = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.illumination_down_blocks = nn.ModuleList([])

        ill_chan = latent_dim

        self.illumination_down_blocks.append(
            nn.Conv2d(in_channels=ill_chan, out_channels=chan, kernel_size=4, stride=4, padding=1))
        ill_chan = chan

        ## optical flow based alignment
        self.flow_matching = RAFT(weights='Raft_Small_Weights.C_T_V2')
        self.flow_matching.eval()
        for param in self.flow_matching.parameters():
            param.requires_grad = False
        
        self.synthesis = nn.Conv2d(in_channels=chan + ill_chan + ill_chan, out_channels=chan,
                                   kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.middle_blks = nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.up_illus.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
            self.de_illus.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        self.fusion = nn.Sequential(
            nn.Conv2d(img_channel * 3, width * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            NONLocalBlock2D(width * 2, sub_sample='bilinear', bn_layer=False),
            nn.Conv2d(width * 2, width * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, 3, 1),
            NONLocalBlock2D(3, sub_sample='bilinear', bn_layer=False)
        )

        self.apply_coeffs = ApplyCoeffs()

    def forward(self, inp):
        ret_dict = {}
        B, N, C, H, W = inp.shape  # torch.Size([16, 5, 3, 512, 960])
        assert N == self.nframes
        padded_inp = self.check_image_size(inp)  # B, C=3, H, W
        B, N, C, pad_H, pad_W = padded_inp.shape  # torch.Size([16, 5, 3, 16x, 16x])        

        # Construct input
        x_center = padded_inp[:, self.center, :, :, :].contiguous()  # B C H W  # [2, 3, 512, 960]
         
        
        illumination, illumination_map = self.illumination_module(x_center)
        illumination_v, illumination_v_map = self.illumination_module(1 - x_center)
        for block in self.illumination_down_blocks:
            illumination = block(illumination)
            illumination_v = block(illumination_v)  
            
        inp = self.align_with_optical_flow(padded_inp, x_center)  # [2, 5, 3, 512, 960]
        

        # 1. calculate average amplitude
        x_amp_freqs = [torch.fft.rfft2(inp[:, i, :, :, :].contiguous(), norm='backward') for i in range(N)]  # list, len(x_amp_freqs)==N
        x_amp_freq = torch.cat(x_amp_freqs, dim=1)
        x_amp_freq_amp = torch.abs(x_amp_freq)
        x_amp_avg = self.amp_map(x_amp_freq_amp)

        x = []
        aligned_frames = []
        # 2. calculate phase
        for i in range(N):
            x_amp_freqs_pha = torch.angle(x_amp_freqs[i])  
            x_amp_pha_new = self.pha_maps[i](x_amp_freqs_pha)
            x_amp_new = x_amp_avg * torch.exp(1j * x_amp_pha_new)
            inp_new = torch.fft.irfft2(x_amp_new, norm='backward').real
            aligned_x = self.fal[i](inp[:, i, :, :, :], inp[:, self.center, :, :, :], x_amp_avg, inp_new)
            aligned_frames.append(aligned_x[:,:,:H,:W])
            x.append(self.frames_Projection[i](aligned_x))
        
        # Add selective IN here for multi-frame input       
        x = torch.stack(x, dim=1).view(B, N * C, pad_H, pad_W)
        
        x, x_norm = self.attin(x)
        x = self.intro(x)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            # BaselineBlock and down: Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2))
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.synthesis(torch.cat([x, illumination, illumination_v], 1))
        
        x = self.middle_blks(x)
        
        # resolution of items in encs is from large to small
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)  # [2, 128, 64, 120]
            x = x + enc_skip
            x = decoder(x)  # [2, 128, 64, 120]->[2, 64, 128, 240]->...->[2, 16, 512, 960]

        x = self.ending(x)  # [2, 3, 512, 960]  # TODO: fix the blur from stacked frames, 0-1

        cated_illus = torch.cat([illumination_map, illumination_v_map], dim=0)  # [4, 16, 512, 960]

        
        image_illus = self.end_illu(cated_illus)  # [4, 12, 512, 960]
        image_illu, image_illu_v = image_illus[:B], image_illus[B:]  # under, overexposed
        image_illu = self.apply_coeffs(image_illu, x_center)  # [2, 3, 512, 960]
        image_illu_v = self.apply_coeffs(image_illu_v, x_center)  # [2, 3, 512, 960]
        
        
        ## compared with the learned image_illu_map, if original pixel values are greater than map, then corresponding pixel approaches 1, otherwise 0 (approximately)
        # find the regions of underexpsoed(brighten_x) and overexposed(darken_x) based on self.decomp
        brighten_x = self.decomp(x_center, image_illu)  # [2, 3, 512, 960]
        darken_x = 1 - self.decomp(1 - x_center, image_illu_v)  # [2, 3, 512, 960]
        fused_x = torch.cat([x, brighten_x, darken_x], dim=1)  # [2, 9, 512, 960]
        # brighten_x: indicate the underexposed
        # darken_x  : indicate the overexposed
        
        weight_map = self.fusion(fused_x)  # [2, 3, 512, 960]
        w1 = weight_map[:, 0, ...].unsqueeze(1)
        w2 = weight_map[:, 1, ...].unsqueeze(1)
        w3 = weight_map[:, 2, ...].unsqueeze(1)
        out = x * w1 + brighten_x * w2 + darken_x * w3
        
        
        
        ret_dict.update({
            'pred': out[:, :, :H, :W],
            'image_illu': image_illu[:, :, :H, :W],
            'image_illu_v': image_illu_v[:, :, :H, :W],
            'brighten_x': brighten_x[:, :, :H, :W],
            'darken_x': darken_x[:, :, :H, :W],            
        })
        return ret_dict

    def warp_by_flow(self, img, flow):
        """
        warp image based on the predicted optical flow
        img:  (B, 3, H, W)
        flow: (B, 2, H, W)
        """
        B, C, H, W = img.size()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=2).float()  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

        # (B, 2, H, W) -> (B, H, W, 2)
        flow = flow.permute(0, 2, 3, 1)
        grid = grid + flow

        grid_x = 2.0 * grid[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid_y = 2.0 * grid[:, :, :, 1] / max(H - 1, 1) - 1.0
        grid_norm = torch.stack((grid_x, grid_y), dim=3)

        warped = F.grid_sample(img, grid_norm, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    def estimate_flow_multiscale(self, img1, img2, scale=0.5):
        """
        img1, img2: (B, 3, H, W)
        scale: to downasample (0.25~0.5)
        """
        B, C, H, W = img1.shape
        small_H, small_W = int(H * scale), int(W * scale)

        img1_small = F.interpolate(img1, size=(small_H, small_W), mode='bilinear', align_corners=False)
        img2_small = F.interpolate(img2, size=(small_H, small_W), mode='bilinear', align_corners=False)

        # on coarse resolution
        flow_small = self.flow_matching(img1_small, img2_small)[-1]  # (B, 2, small_H, small_W)

        # 
        flow = F.interpolate(flow_small, size=(H, W), mode='bilinear', align_corners=False)
        flow = flow * (1.0 / scale)

        return flow

    def align_with_optical_flow(self, frames, x_center):
        """
        frames: (B, N, 3, H, W)
        x_center: (B, 3, H, W)
        """
        B, N, C, H, W = frames.shape
        center = N // 2

        aligned_frames = []
        for i in range(N):
            if i == center:
                aligned_frames.append(x_center)
            else:
                # current frame -> x_center
                flow = self.estimate_flow_multiscale(frames[:, i, :, :, :], x_center)  # (B, 2, H, W)
                # warp current frame
                warped = self.warp_by_flow(frames[:, i, :, :, :], flow)  # B, 3, H, W
                aligned_frames.append(warped)

        # stack tensor (B, N, 3, H, W)
        aligned_frames = torch.stack(aligned_frames, dim=1)
        return aligned_frames

    def decomp(self, x1, illu_map):
        # denominator: returns the larger values between illu_map and x1
        return x1 / (torch.where(illu_map < x1, x1, illu_map.float()) + 1e-7)

    def check_image_size(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
