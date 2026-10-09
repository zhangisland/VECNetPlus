from loguru import logger
from torch.nn.modules.loss import _Loss

from yacs.config import CfgNode as CN
from loguru import logger
from losses.image_loss import (
    CB_SSIM_Loss,
    TV_Loss,
    L1_Loss,
    SSIMBasic,
    COLOR_Loss,
    MASK_Loss,
    EXC_Loss, # contrastive color correction loss
    ColorConstancyFastLoss,
    ColorConstancyZeroDCELoss,
    L1CharbonnierLoss,
    SobelLoss,
    AMP_Loss,
)

from losses.loss_registry import build_loss


def get_single_loss(name, **kwargs):
    return build_loss(name, **kwargs)

class MixedLoss:
    def __init__(self, cfg):
        """        
        LOSSES:
        SINGLE: false
        L1CharbonnierLoss:
            weight: 1.0
            enable: true
        TV_LOSS:
            weight: 0.5
            enable: true
        ...
        """
        self.loss_items = {}

        for loss_name, loss_cfg in cfg.items():
            
            if loss_name == "SINGLE" and loss_cfg:
                self.loss_items['L1CharbonnierLoss'] = {
                    "fn": build_loss('L1CharbonnierLoss'),
                    "weight": 1.0,
                }
                break
                    
            if not isinstance(loss_cfg, dict) or not loss_cfg.get("enable", False):
                continue

            weight = loss_cfg.get("weight", 1.0)
            loss_fn = build_loss(loss_name)

            self.loss_items[loss_name] = {
                "fn": loss_fn,
                "weight": weight,
            }
    def __call__(self, batch, **kwargs):
        total = 0
        details = {}
        for name, item in self.loss_items.items():
            loss = item["fn"](batch, **kwargs)
            weighted = loss * item["weight"]
            total += weighted
            details[name] = weighted #.item()

        return total, details
