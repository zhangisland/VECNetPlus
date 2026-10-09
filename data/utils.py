import torch
import cv2
import numpy as np
import os.path as osp
from glob import glob
import random
from loguru import logger

def check_shape(data_dict):
    """
    Check the shape of the data in the data_dict.
    """
    logger.debug('Data shape:')
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            logger.debug(f'{k}: {v.shape}')
        elif isinstance(v, np.ndarray):
            logger.debug(f'{k}: {v.shape}')
        elif isinstance(v, list):
            logger.debug(f'{k}: {len(v)}')
        else:
            logger.debug(f'{k}: {type(v)}')
    logger.debug('END')
    return


def read_single_img(env, path, size=None):
    """
    Read image from path.
    env: str, environment name
    path: str, image path
    size: tuple, (H, W)
    """
    if env is None:  # img
        if path.endswith('.npy'):
            img = np.load(path).astype(np.float32)  # H,W,C  BGR
        elif path.lower().endswith(('.tif', '.tiff')):
            try:
                from PIL import Image
                pil_img = Image.open(path)
                if pil_img.mode in ['I;16', 'I;16B', 'I;16L']:
                    img_array = np.array(pil_img)
                    if img_array.dtype == np.uint16:
                        img_array = (img_array / 256).astype(np.uint8)
                    img = img_array
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=2)
                else:
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    img = np.array(pil_img)
                # RGB -> BGR
                if img.ndim == 3 and img.shape[2] == 3:
                    img = img[:, :, ::-1].copy()
            except:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = img.astype(np.float32)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # H,W,C  BGR
        

        if img is None:
            raise ValueError(f'Fail to read {path}, None returned')
        if size is not None:
            h, w = img.shape[:2]
            if (h, w) != size:
                img = cv2.resize(img, (size[1], size[0]))  # WxH
        
    else:
        raise NotImplementedError(f'Unsupported env type: {env}')
        
    # padding to ensure 16x resolution
    # h, w = img.shape[:2]
    # pad_factor = 16
    # pad_h = (pad_factor - h % pad_factor) % pad_factor
    # pad_w = (pad_factor - w % pad_factor) % pad_factor
    # if pad_h > 0 or pad_w > 0:
    #     img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    # end padding

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_img_seq(path, size=None):
    """
    read images from a folder or a list of image paths.
    path: str, image path
    size: tuple, (H, W)
    return: imgs, torch tensor, (N, C, H, W)
    """
    if type(path) is list:
        img_path_list = path
    else:
        img_path_list = sorted(glob(osp.join(path, '*')))

    img_list = [read_single_img(None, v, size) for v in img_path_list]
    # stack to Torch tensor
    imgs = np.stack(img_list, axis=0)
    try:
        imgs = imgs[:, :, :, [2, 1, 0]]  # BGR to RGB
    except Exception:
        import pdb; pdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def random_crop(img_gts, img_lqs, patch_size, scale=1, stage='train'):
    """random crop.
    It crops lists of lq/gt with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        
        patch_size (tuple(int)): cropped patch size: H,W.
        scale (int): Scale factor.
        stage (str): 'train' or 'validate'.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    
    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    
    assert h_lq==h_gt and w_lq==w_gt, 'The size of lq image and gt image mismatch.'

    lq_patch_size_h, lq_patch_size_w = patch_size[0] // scale, patch_size[1] // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size_h or w_lq < lq_patch_size_w:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                        f'({lq_patch_size_h}, {lq_patch_size_w}). ')


    # randomly choose top and left coordinates for lq patch
    if stage == 'train':
        top = random.randint(0, h_lq - lq_patch_size_h)
        left = random.randint(0, w_lq - lq_patch_size_w)
    else:
        # centered crop
        top = (h_lq - lq_patch_size_h) // 2
        left = (w_lq - lq_patch_size_w) // 2
    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size_h, left:left + lq_patch_size_w, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + patch_size[0], left_gt:left_gt + patch_size[1], ...]
        for v in img_gts
    ]
    imgs_ret = img_gts
    imgs_ret.extend(img_lqs)

    return imgs_ret


def random_crop_torch(img_list, patch_size, stage='train'):
    """random crop.
    It crops lists of lq/gt with corresponding locations.

    Args:
        img_list (list[torch.Tensor] | torch.Tensor): [GT, LQs...] images. Note that all images
            should have the same shape. If the input is a tensor, it will
            be transformed to a list containing itself.
            shape: [number of images, channel, height, width]
        patch_size (tuple(int)): cropped patch size: H,W.
        stage (str): 'train' or 'validate'.

    Returns:
        list[torch.Tensor] | torch.Tensor: GT images and LQ images. If returned results
            only have one element, just return tensor.
    """

    if not isinstance(img_list, list):
        img_list = [img_list]
    
    _, h_gt, w_gt = img_list[0].shape
    patch_size_h, patch_size_w = patch_size[0], patch_size[1]

    if stage == 'train':
        top = random.randint(0, h_gt - patch_size_h)
        left = random.randint(0, w_gt - patch_size_w)
    else:
        # centered crop
        top = (h_gt - patch_size_h) // 2
        left = (w_gt - patch_size_w) // 2
    # crop lq patch
    img_list = [
        v[..., top:top + patch_size_h, left:left + patch_size_w]
        for v in img_list
    ]
    
    return img_list


# data augmentation
def augment(img_list, use_flip=True, use_rot=True):
    """
    img_list: list of numpy arrays, (3, H, W)
    """
    hflip = use_flip and random.random() < 0.5
    vflip = use_flip and random.random() < 0.5
    rot90 = use_rot and random.random() < 0.5
    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    img_list = [_augment(img) for img in img_list]
    return img_list

def augment_torch(img_list, use_flip=True, use_rot=True):
    """
    img_list: list of torch tensors, (N, C, H, W)
    """
    hflip = use_flip and random.random() < 0.5
    vflip = use_flip and random.random() < 0.5
    rot90 = use_rot and random.random() < 0.5
    def _augment(img):
        if hflip:
            img = torch.flip(img, dims=[2])
        if vflip:
            img = torch.flip(img, dims=[1])
        if rot90:
            img = torch.flip(torch.transpose(img, 1, 2), dims=[1])  # clockwise rot
        return img
    img_list = [_augment(img) for img in img_list]
    return img_list

def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

