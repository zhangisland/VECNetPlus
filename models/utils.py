import os
import os.path as osp
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid
from matplotlib import cm



def rgb2yCbCr(RGB):
    Y = 0.2990 * RGB[:, 0, :, :] + 0.5870 * RGB[:, 1, :, :] + 0.1140 * RGB[:, 2, :, :]
    Y1 = Y
    Y = Y.reshape(Y.shape[0], 1, Y.shape[1], Y.shape[2])
    U = (RGB[:, 2, :, :] - Y1) / 1.772
    U = U.reshape(U.shape[0], 1, U.shape[1], U.shape[2])
    V = 0.5 * RGB[:, 0, :, :] - 0.419 * RGB[:, 1, :, :] - 0.081 * RGB[:, 2, :, :]
    V = V.reshape(V.shape[0], 1, V.shape[1], V.shape[2])
    YUV = torch.cat([Y, U, V], 1)
    return YUV

def compute_residual_map(output, frame):
    """
    Compute absolute residual map |output - frame|.
    Input tensors must be normalized to [0, 1].
    Shape: [B, C, H, W]
    """
    return torch.abs(output - frame).mean(dim=1, keepdim=True)  # -> [B, 1, H, W]

# Create folder
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def normalize_map(x, eps=1e-6):
    # x: torch tensor (B, H, W) or (H, W)
    x = x.float()
    mn = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] if x.dim() == 3 else x.min()
    mx = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] if x.dim() == 3 else x.max()
    return (x - mn) / (mx - mn + eps)

def single_channel_to_heatmap(img_np, cmap='viridis'):
    # img_np: HxW values in [0,1]
    cmap = cm.get_cmap(cmap)
    colored = cmap(img_np)[:, :, :3]  # H W 3
    return (colored * 255).astype(np.uint8)

def overlay_heatmap_on_image(img, heatmap, alpha=0.5):
    # img: H W 3 uint8 (0-255), heatmap: H W 3 uint8
    return (img.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha).astype(np.uint8)

def save_image(np_img, path):
    plt.imsave(path, np_img)


def save_feature_map_grid(self, feat, path, ncols=8, cmap='viridis', per_channel=True):
    """
    feat: torch tensor (C, H, W) or (B, C, H, W) or (H, W)
    If per_channel==True and feat has channels, saves grid of single-channel heatmaps.
    """
    ensure_dir(os.path.dirname(path))
    if feat.dim() == 4:
        feat = feat[0]  # first item in batch
    if feat.dim() == 3:
        C, H, W = feat.shape
        # normalize each channel
        channels = []
        for c in range(C):
            m = normalize_map(feat[c]).numpy()
            hm = single_channel_to_heatmap(m)
            channels.append(hm)
        # make grid image manually (ncols)
        ncols = min(ncols, C)
        nrows = (C + ncols - 1) // ncols
        pad_h = nrows * H
        pad_w = ncols * W
        grid = np.zeros((pad_h, pad_w, 3), dtype=np.uint8) + 0
        for idx, ch in enumerate(channels):
            r = idx // ncols
            c = idx % ncols
            grid[r*H:(r+1)*H, c*W:(c+1)*W] = ch
        save_image(grid, path)
    elif feat.dim() == 2:
        m = normalize_map(feat).numpy()
        save_image(single_channel_to_heatmap(m), path)
    else:
        # single 3-channel image HWC or tensor (3,H,W)
        if feat.dim() == 3 and feat.shape[0] == 3:
            img = feat.permute(1,2,0).numpy()
            img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-6), 0, 1)
            save_image((img * 255).astype(np.uint8), path)

def visualize_flow_magnitude(self, flow, path):
    # flow: (B,2,H,W) -> save magnitude heatmap
    if flow.dim() == 4:
        flow = flow[0]
    mag = torch.sqrt(flow[0]**2 + flow[1]**2)
    mag_n = normalize_map(mag).numpy()
    save_image(single_channel_to_heatmap(mag_n), path)
