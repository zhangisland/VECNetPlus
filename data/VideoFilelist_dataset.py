import numpy as np
import os
import os.path as osp

from torch.utils.data import Dataset, ConcatDataset
import cv2
import random
import torch

from data.utils import (
    img2tensor, 
    index_generation, 
    read_img_seq, 
    random_crop_torch,
    augment_torch,
    check_shape,
)
from glob import glob
from loguru import logger

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".npy", ".tif", ".tiff"])

class VideoFilelistDataset(Dataset):
    def __init__(self, 
                 lq_seq,
                 gt_seq,
                 exposure_label,
                 height,
                 width,
                 seq_name,        
                 patch_size,
                 n_frames,
                 cache_data,
                 padding,
                 stage,
                 use_flip=False,
                 use_rot=False,
                 random_reverse=False,
                 fast_validation=False,
                 fast_validation_frame=30,
                 keep_original_resolution=False):
        super().__init__()
        
        # logger.info(f'Creating {__class__.__name__} dataset with {seq_name}')
        self.H = height
        self.W = width
        self.n_frames = n_frames
        self.cache_data = cache_data
        self.padding = padding
        self.stage = stage
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.random_reverse = random_reverse
        self.fast_validation = fast_validation
        self.fast_validation_frame = fast_validation_frame
        self.keep_original_resolution = keep_original_resolution
        self.data_info = {
            'border': [],
            'index': [],  # idx of current frame / number of frames in this sequence
            'exposure_label': []
        }
        self.gt_img_dir = gt_seq
        self.gt_img_list = sorted(glob(osp.join(gt_seq, '*')))
        self.gt_img_list = [x for x in self.gt_img_list if is_image_file(x)]
                
        self.lq_img_list = sorted(glob(osp.join(lq_seq, '*')))
        self.lq_img_list = [x for x in self.lq_img_list if is_image_file(x)]
        if self.fast_validation:
            min_len = min(len(self.gt_img_list), self.fast_validation_frame)
            self.gt_img_list = self.gt_img_list[:min_len]
            self.lq_img_list = self.lq_img_list[:min_len]

        if patch_size is None:
            self.random_cropped_height = self.H
            self.random_cropped_width = self.W
        else:
            self.random_cropped_height, self.random_cropped_width = patch_size
        self.seq_name = seq_name  # seqname folder, postfix with '/'
        max_idx = len(self.lq_img_list)
        
        border_l = [0] * max_idx
        for i in range(self.n_frames//2):
            border_l[i] = 1
            border_l[max_idx - i - 1] = 1
        self.data_info['border'] = border_l
        for i in range(max_idx):
            self.data_info['index'].append(f'{i:05d}/{max_idx:05d}')  # idx of current frame / number of frames in this sequence
            self.data_info['exposure_label'].append(exposure_label)
        # logger.info(f'Loading {self.seq_name} with {len(self.lq_img_list)} frames')
                

    def __getitem__(self, index):        
        idx, max_idx = self.data_info['index'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        select_idx = index_generation(crt_i=idx,
                                      max_n=max_idx,
                                      N=self.n_frames,
                                      padding=self.padding)
        lq_imgs_path = []
        for mm in range(len(select_idx)):
            lq_imgs_path.append(self.lq_img_list[select_idx[mm]])
        gt_img_path = self.gt_img_list[index:index+1]
        if self.stage != 'test':
            lq_imgs = read_img_seq(lq_imgs_path, size=(self.H, self.W))  # normalized to [0, 1]
            gt_img = read_img_seq(gt_img_path, size=(self.H, self.W))  # read original size first, [N,C,H,W]
        else:
            if self.keep_original_resolution:
                lq_imgs = read_img_seq(lq_imgs_path, size=None)  # normalized to [0, 1]
                gt_img = read_img_seq(gt_img_path, size=None)  # read original size first, [N,C,H,W]
            else:
                lq_imgs = read_img_seq(lq_imgs_path, size=(self.H, self.W))  # normalized to [0, 1]
                gt_img = read_img_seq(gt_img_path, size=(self.H, self.W))  # read original size first, [N,C,H,W]

        

        gt_img = gt_img[0]
        lq_imgs = list(lq_imgs.unbind(0))  # to list
        # gt_img: torch.Size([3, 512, 960]), len(lq_imgs): 5, lq_imgs[0]: torch.Size([3, 512, 960])
        img_list = [gt_img]
        img_list.extend(lq_imgs)
        # logger.debug(f'len(img_list): {len(img_list)}, img_list[0].shape={img_list[0].shape}')
        # len(img_list): 6, img_list[0].shape=torch.Size([3, 512, 960])
        if self.stage != 'test':
            # crop
            imgs_ret = random_crop_torch(
                img_list=img_list,
                patch_size=(self.random_cropped_height, self.random_cropped_width),
                stage=self.stage,
            )  # [gt, lq1, lq2, ...]
            # augment
            if self.stage == 'train':
                imgs_ret = augment_torch(img_list=imgs_ret,
                                         use_flip=self.use_flip,
                                         use_rot=self.use_rot)
            gt_img = imgs_ret[0]
            lq_imgs = imgs_ret[1:]
        lq_imgs = torch.stack(lq_imgs, dim=0)  # (N, C, H, W)        

        sample = {
            "lqs": lq_imgs,  # [n_frames, C, H, W]
            "gt": gt_img, # [C, H, W]
            "seq_name": self.seq_name,
            "lq_path_list": lq_imgs_path,
            "gt_path": gt_img_path[0],
            'exposure_label': self.data_info['exposure_label'][index]
        }
        # check_shape(data_dict=sample)

        return sample

    def __len__(self):
        return len(self.gt_img_list)

def get_videofilelist_dataset(
    stage,
    config,
    keep_original_resolution=False,
):
    if stage not in ['train', 'validate', 'test']:
        raise ValueError(f"Invalid stage: {stage}. Available stages: ['train', 'validate', 'test']")

    params = [getattr(config.DATA[stage], param)
             for param in ["dataroot_txt", "img_size", "patch_size", "N_frames", "cache_data", "padding"]]
    try:
        dataroot_txt, (height, width), patch_size, n_frames, cache_data, padding = params
    except:
        import pdb; pdb.set_trace()
        raise ValueError(f"Invalid config.DATA.{stage} parameters.")
    dataset_name = osp.basename(dataroot_txt)[:-4]
    
    if stage == 'train':
        use_flip = config.DATA.train.use_flip
        use_rot = config.DATA.train.use_rot
        random_reverse = config.DATA.train.random_reverse
    else:
        use_flip = False
        use_rot = False
        random_reverse = False

    stage = stage
    groups = [line.rstrip() for line in open(dataroot_txt)]
    group_filenames = [group.split('|') for group in groups]

    if config.SYSTEM.debug:
        group_filenames = group_filenames[:2]
        
    logger.info(f'Using {dataroot_txt}[#len={len(group_filenames)}]: {group_filenames[0]}...')

    seq_dataset_list = []
    for idx, group in enumerate(group_filenames):
        # lq_seq|gt_seq|exposure_label
        lq_seq, gt_seq, exposure_label = group
        if 'realhdrv' in dataset_name.lower() or 'lliv' in dataset_name.lower() or 'bdd' in dataset_name.lower():
            format_seq_name = os.sep.join(lq_seq.split(os.sep)[-2:])
        else:
            format_seq_name = osp.basename(lq_seq)
        
        seq_dataset_list.append(
            VideoFilelistDataset(
                lq_seq=lq_seq,
                gt_seq=gt_seq,
                exposure_label=exposure_label,
                height=height,
                width=width,
                seq_name=format_seq_name,
                patch_size=patch_size,
                n_frames=n_frames,
                cache_data=cache_data,
                padding=padding,
                stage=stage,
                use_flip=use_flip,
                use_rot=use_rot,
                random_reverse=random_reverse,
                fast_validation=config.SYSTEM.fast_validation,
                keep_original_resolution=config.DATA.test.keep_original_resolution
            )
        )

    return ConcatDataset(seq_dataset_list)


