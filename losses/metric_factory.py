from loguru import logger
from torch import nn
from yacs.config import CfgNode as CN
from losses.image_loss import PSNRMetric, SSIMMetric


def get_single_metric(name):
    if name == "SSIM":
        return SSIMMetric()
    elif name == "PSNR":
        return PSNRMetric()
    else:
        raise ValueError(f"Unknown metric config: {name}")


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.metric = []
        self.eval_func = []
        for name in configs.keys():
            cfg_node = getattr(configs, name)
            if isinstance(cfg_node, CN):
                self.metric.append(name)
                self.eval_func.append(get_single_metric(name))

    def forward(self, x, y):
        r = []
        for m, e in zip(self.metric, self.eval_func):
            r.append((m, e(x.detach(), y.detach())))
        return r
