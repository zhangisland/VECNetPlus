import math
import torch.nn as nn
from models import load_arch


class VECNetAlignEnhance(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg  # config.MODEL
        
        EnhanceNet = load_arch(self.cfg)
        self.model = EnhanceNet(down_blocks=cfg.BASE.down_blocks,
                                width=cfg.BASE.width,
                                nframes=cfg.BASE.nframes,
                                middle_blk_num=cfg.BASE.middle_blk_num,
                                enc_blk_nums=cfg.BASE.enc_blk_nums,
                                dec_blk_nums=cfg.BASE.dec_blk_nums,
                                dw_expand=cfg.BASE.dw_expand,
                                ffn_expand=cfg.BASE.ffn_expand,
                                **kwargs)
        
    def forward(self, batch):
        result = self.model(batch['lqs'])

        result.update({
            'aligned_frames': result.get('aligned_frames', None),
            'gt': batch['gt'] if 'gt' in batch else None,
            'lq': batch['lqs'][:, math.floor(self.cfg.BASE.nframes/2), ...],
        })
        return result
