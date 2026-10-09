import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import sys
import pandas as pd
import argparse
from loguru import logger


parser = argparse.ArgumentParser(description="DIMEPlus dataset configuration")

parser.add_argument(
    "--local_dataset_path",
    type=str,
    default="datasets/DIMEPlus/test/LQ",
    help="Path to the DIMEPlus dataset"
)

parser.add_argument("--phase", type=str, choices=["train", "valid", "test"], default="test")
parser.add_argument("--split_output", action='store_true', default=False, help='true if save to LQ|GT.txt')
parser.add_argument("--delimiter", type=str, default='|', choices=[',', '|'])


args = parser.parse_args()

DATASET_PATH = args.local_dataset_path 
DATASET_PATH = osp.abspath(DATASET_PATH)

output_txt_folder = 'dataPath'
if not osp.exists(output_txt_folder):
    os.makedirs(output_txt_folder)


prepare_type = args.phase
scene_type = 'DIMEPlus'
delimiter = args.delimiter


if prepare_type in ['train', 'test', 'valid']:
    seq_list = sorted((glob(osp.join(DATASET_PATH, '*'))))
    print(f'#seq={len(seq_list)}:\n{seq_list}')
    if not args.split_output:
        f_lq = open(osp.join(output_txt_folder, f"{scene_type}_{prepare_type}.txt"), 'w')
    else:
        f_lq = open(osp.join(output_txt_folder, f"{scene_type}_{prepare_type}_LQ.txt"), 'w')
        f_gt = open(osp.join(output_txt_folder, f"{scene_type}_{prepare_type}_GT.txt"), 'w')


    for seq_idx, seq in tqdm(enumerate(seq_list), total=len(seq_list)):
        # datasets/DIMEPlus/np/train/LQ/001
        # datasets/DIMEPlus/np/train/GT/001
        lq_seq = seq        
        gt_seq = osp.join(osp.dirname(DATASET_PATH), 'GT', osp.basename(seq))
        
        if not args.split_output:
            f_lq.write(f'{lq_seq}{delimiter}{gt_seq}{delimiter}-1\n')
        else:
            f_lq.write(f'{lq_seq}\n')
            f_gt.write(f'{gt_seq}\n')
    f_lq.close()
    if args.split_output:
        f_gt.close()
else:
    print("Wrong dataset type.")

