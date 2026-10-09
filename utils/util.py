import os
import os.path as osp
import re
from glob import glob
from loguru import logger
import torch
import shutil
import hashlib
import cv2
from PIL import Image
import numpy as np

import math

from torchvision.utils import make_grid

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model

def gen_expid(instr):
    return hashlib.md5(instr.encode('utf-8')).hexdigest()[:8]


def create_exp_dir(path, scripts_to_save=None):
    """create a experiment directory to save logs, codes, models, etc."""
    if scripts_to_save is None:
        logger.waring('No script to save.')
        return
    
    os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
    logger.info(f'Experiment dir : {path}')
    for script in scripts_to_save:
        dst_file = os.path.join(path, 'scripts', os.path.basename(script))
        if os.path.isdir(script):
            shutil.copytree(script, dst_file, dirs_exist_ok=True)
        else:
            shutil.copyfile(script, dst_file)

def mkdir_for_config(config):
    for key, value in config.SYSTEM.items():
        if 'dir' in key and value is not None and value != '':
            os.makedirs(value, exist_ok=True)
    
def save_pred_img(pred, save_path):
    """
    convert predicted <tensor> to image and save it to the specified directory
    """
    pred = pred.cpu().numpy()
    pred = np.transpose(pred, (1, 2, 0))
    pred = (pred * 255).astype(np.uint8)
    cv2.imwrite(save_path, pred)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)



def save_ckpt_epoch(model, optimizer, lr_scheduler, epoch, num_steps, save_path, max_num_ckpts=2):
    """
    save model, optimizer, lr_scheduler, and epoch to the specified directory
    max_num_ckpts: maximum number of checkpoints to keep
    """
    ckpt_dir = osp.dirname(save_path)
    ckpts = sorted(osp.join(ckpt_dir, x) for x in os.listdir(ckpt_dir) if x.endswith('.pth') and 'best' not in x)
    if len(ckpts) >= max_num_ckpts:
        os.remove(ckpts[0])
    torch.save({
        'checkpoint': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'num_steps': num_steps,
    }, save_path)


def save_final_ckpt(model, save_path):
    torch.save(model.state_dict(), save_path)


def save_best_ckpt(model, optimizer, lr_scheduler, epoch, num_steps, save_path, metric_name, reverse=True, num_keep=3):
    """
    save best_model based on metric value
    reverse: True for psnr/ssim, False for loss    
    """
    torch.save({
        'checkpoint': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'num_steps': num_steps,
    }, save_path)
    # check existing ckpts
    existing_ckpts = sorted(glob(osp.join(osp.dirname(save_path), f'*_best{metric_name}=*.pth')))
    parsed_ckpts = []
    # Parse metric values from filenames
    for ckpt in existing_ckpts:
        ckpt_name = osp.basename(ckpt)
        match = re.search(fr'best{re.escape(metric_name)}=([+-]?\d+\.\d+)\.pth', ckpt_name)
        if match:
            value = float(match.group(1))
            parsed_ckpts.append((value, ckpt))
    # Sort checkpoints by metric value, True for psnr/ssim, False for loss
    sorted_ckpts = [ckpt for (val, ckpt) in sorted(parsed_ckpts, reverse=reverse)]

    # Keep top num_keep and remove others
    if len(sorted_ckpts) > num_keep:
        for ckpt in sorted_ckpts[num_keep:]:
            os.remove(ckpt)


def analysis_seqname_from_filepath(filepath):
    parts = filepath.split(os.sep)
    if 'dime' in filepath.lower():
        seqname = parts[-2]
    else:
        seqname = parts[-2]
    return seqname