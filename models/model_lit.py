import lightning.pytorch as pl
import cv2

from torch.optim import lr_scheduler
from torch.optim import SGD, Adam, AdamW

import os
import os.path as osp
import numpy as np
from models.utils import ensure_dir, save_image, single_channel_to_heatmap, normalize_map
from loguru import logger
import time
import torch

from utils.util import tensor2img, analysis_seqname_from_filepath
from yacs.config import CfgNode as CN
from losses import AverageMeter, get_loss, get_metric

from torchinfo import summary
from models import custom_lr_scheduler
import torchvision
from collections import defaultdict

from torch.cuda.profiler import profile
from torch.profiler import profile, record_function, ProfilerActivity


from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # 1. Build main model
        if self.config.MODEL.model_name == 'VECNetPlus':
            from models.vecnet_enhance import VECNetAlignEnhance
            self.model = VECNetAlignEnhance(config.MODEL)
        else:
            raise ValueError(f'{config.MODEL.model_name} not defined')
        # 2. Build loss function with detailed logging        
        self.criterion = get_loss(self.config.LOSSES)

        # 3. Build metrics with detailed logging
        self.metrics = get_metric(self.config.METRICS)
        # 4. Initialize other attributes
        self.losses_meter = None
        self.metric_meter = None
        self.val_losses_meter = None
        self.val_metric_meter = None
        self.test_losses_meter = None
        self.test_metric_meter = None
        self.train_dataloader_length = None
        self.val_dataloader_length = None
        self.nframes = self.config.MODEL.BASE.nframes
        
        # complexity statistical
        self.pure_inference_time = 0.0
        
        self.is_inference = True if self.config.TEST.test else False
        logger.info(f'self.is_inference: {self.is_inference}')

    def forward(self, x):
        if self.is_inference:
            torch.cuda.synchronize()
            start = time.time()

        out = self.model(x)

        if self.is_inference:
            torch.cuda.synchronize()
            self.pure_inference_time += (time.time() - start)
        return out

    def _format_log(self, losses_meter, metric_meter):
        log_str = 'Loss ='
        for i, (name, meter) in enumerate(losses_meter.items()):
            sep = " [ " if i == 0 else ", "
            log_str += f"{sep}{name}: {meter.avg:.6e}"
        log_str += ' ], Metric ='
        for i, (name, measure) in enumerate(metric_meter.items()):
            sep = " [ " if i == 0 else ", "
            log_str += f"{sep}{name}: {measure.avg:.6f}"
        log_str += ' ]'
        return log_str
    
    def on_fit_start(self):
        self.train_dataloader_length = self.trainer.num_training_batches
        self.val_dataloader_length = self.trainer.num_val_batches[0] if self.trainer.num_val_batches else 0
        self.total_steps = min(self.config.TRAIN.num_epochs * self.train_dataloader_length, self.config.TRAIN.num_steps)
        logger.info(f'Model arch:\n{summary(self.model, verbose=0, depth=2)}')
        # logger.info(f'Model arch:\n{self.model}')
        self._init_meters()
        
    def _init_meters(self):
        """
        avoid resuming from step, leading to missing init self.losses_meter (returned None)
        """
        self.losses_meter = {"TotalLoss": AverageMeter("Train/TotalLoss")}
        if self.config.LOSSES.SINGLE:
            self.losses_meter['PIXEL_LOSS'] = AverageMeter("Train/PIXEL_LOSS")
        else:
            for name in self.config.LOSSES.keys():
                cfg_node = getattr(self.config.LOSSES, name)
                if isinstance(cfg_node, CN) and hasattr(cfg_node, 'weight') and getattr(cfg_node, 'enable', False):
                    self.losses_meter[name] = AverageMeter(f"Train/{name}")

        self.metric_meter = {}
        for name in self.config.METRICS.keys():
            cfg_node = getattr(self.config.METRICS, name)
            if isinstance(cfg_node, CN) and hasattr(cfg_node, 'only_Y'):
                self.metric_meter[name] = AverageMeter(f"Train/{name}")

    def on_train_epoch_start(self):
        for m in self.losses_meter.values():
            m.reset()
        for m in self.metric_meter.values():
            m.reset()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        
        if not hasattr(self, "losses_meter") or self.losses_meter is None:
            self._init_meters()


        outputs = self.model(batch)  # return updated batch items
        losses, name_to_loss = self.criterion(outputs)
        name_to_measure = self.metrics(x=outputs['pred'], y=outputs['gt'])  # PSNR, SSIM
        
        self.losses_meter["TotalLoss"].update(losses.detach().item())
        for name, loss_item in name_to_loss.items():
            self.losses_meter[name].update(loss_item.detach().item())
        for name, measure_item in name_to_measure:
            if isinstance(measure_item, torch.Tensor):
                measure_item = measure_item.detach().item()
            self.metric_meter[name].update(measure_item)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.log('total_loss', self.losses_meter["TotalLoss"].avg, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_PSNR', self.metric_meter["PSNR"].avg, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_SSIM', self.metric_meter["SSIM"].avg, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.global_step % self.config.TRAIN.print_freq == 0:
            log_str = f'Train Epoch {self.current_epoch:03d}/{self.config.TRAIN.num_epochs:03d} | Step {self.global_step:06d}/{self.total_steps:06d}, ' + self._format_log(self.losses_meter, self.metric_meter)
            logger.info(log_str)
        
        return losses
    
    def on_train_epoch_end(self):
        log_str = f'Train Epoch {self.current_epoch:03d}/{self.config.TRAIN.num_epochs:03d} | Step {self.global_step:06d}/{self.total_steps:06d}' + self._format_log(self.losses_meter, self.metric_meter)
        logger.info(log_str)
        self.log('train_loss', self.losses_meter["TotalLoss"].avg, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.TRAIN.batch_size)

        for metric_name, metric_value in self.metric_meter.items():
            self.log(f'train_{metric_name}', metric_value.avg, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.TRAIN.batch_size)
    
    ## init val meter
    def on_validation_start(self):
        self.val_losses_meter = {"TotalLoss": AverageMeter("Val/TotalLoss")}
        if self.config.LOSSES.SINGLE:
            self.val_losses_meter['PIXEL_LOSS'] = AverageMeter("Val/PIXEL_LOSS")
        else:
            for name in self.config.LOSSES.keys():
                cfg_node = getattr(self.config.LOSSES, name)
                if isinstance(cfg_node, CN) and hasattr(cfg_node, 'weight') and getattr(cfg_node, 'enable', False):
                    self.val_losses_meter[name] = AverageMeter(f"Val/{name}")
        self.val_metric_meter = {}
        for name in self.config.METRICS.keys():
            cfg_node = getattr(self.config.METRICS, name)
            if isinstance(cfg_node, CN) and hasattr(cfg_node, 'only_Y'):
                self.val_metric_meter[name] = AverageMeter(f"Val/{name}")

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        if batch_idx == 0:
            # save the first predicted image to check
            pred_image = outputs['pred']  # NCHW
            pred_image = torch.clamp(pred_image, 0, 1)
            if pred_image.dim() == 4:
                pred_image = pred_image[0]
            saved_pred_path = osp.join(self.config.SYSTEM.log_dir, 'validate_visualization', f'epoch{self.current_epoch:03d}_step{self.global_step:08d}_val_{batch_idx:03d}_pred.png')
            if not osp.exists(osp.dirname(saved_pred_path)):
                os.makedirs(osp.dirname(saved_pred_path))

            torchvision.utils.save_image(pred_image, saved_pred_path)


        val_losses, val_name_to_loss = self.criterion(outputs)
        val_name_to_measure = self.metrics(x=outputs['pred'], y=outputs['gt'])
        try:
            self.val_losses_meter["TotalLoss"].update(val_losses.detach().item())
        except:
            import pdb; pdb.set_trace()
        for name, loss_item in val_name_to_loss.items():
            loss_item = loss_item.detach().item()
            self.val_losses_meter[name].update(loss_item)
        for name, measure_item in val_name_to_measure:
            measure_item = measure_item.detach().item()
            self.val_metric_meter[name].update(measure_item)

        
        return val_losses
        
    def on_validation_epoch_end(self):
        log_str = f'Val Epoch {self.current_epoch:03d}/{self.config.TRAIN.num_epochs:03d} | Step {self.global_step:06d}/{self.total_steps:06d} ' + self._format_log(self.val_losses_meter, self.val_metric_meter)
        logger.info(log_str)

        self.log('val_loss', self.val_losses_meter["TotalLoss"].avg, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.VALIDATE.batch_size)

        for metric_name, metric_value in self.val_metric_meter.items():
            self.log(f'val_{metric_name}', metric_value.avg, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.VALIDATE.batch_size)
        
        return super().on_validation_epoch_end()


    def configure_optimizers(self):
        optimizer_cfg = self.config.TRAIN.OPTIMIZER
        # 1. create optimizer
        # config.TRAIN.OPTIMIZER
        if optimizer_cfg.NAME == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=optimizer_cfg.LR, weight_decay=optimizer_cfg.get('weight_decay', 0))
        elif optimizer_cfg.NAME == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=optimizer_cfg.LR, betas=(optimizer_cfg.beta1, optimizer_cfg.beta2), weight_decay=optimizer_cfg.get('weight_decay', 0.01))
        elif optimizer_cfg.NAME == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=optimizer_cfg.LR, weight_decay=optimizer_cfg.get('weight_decay', 0))
        else:
            raise ValueError(f"Unknown Optimizer config: {optimizer_cfg}")
        
        # 2. create scheduler        
        if optimizer_cfg.LR_SCHEDULER == "multi_step":
            self.lr_scheduler = lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=optimizer_cfg.milestones,
                gamma=optimizer_cfg.decay_gamma,
            )
        elif optimizer_cfg.LR_SCHEDULER == "cosine":
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, T_max=optimizer_cfg.end_epoch, eta_min=optimizer_cfg.eta_min)
        elif optimizer_cfg.LR_SCHEDULER == "cosine_w":
            # customized lr scheduler            
            self.lr_scheduler = custom_lr_scheduler.CosineAnnealingLR_Restart(
                    optimizer=self.optimizer, T_period=optimizer_cfg.T_period, 
                    restarts=optimizer_cfg.restarts, weights=optimizer_cfg.restart_weights,
                    eta_min=optimizer_cfg.eta_min)
            # self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                # optimizer=self.optimizer, T_0=optimizer_cfg.restart_T, T_mult=optimizer_cfg.restart_weights, eta_min=optimizer_cfg.eta_min)
        elif optimizer_cfg.LR_SCHEDULER == "onecycle":
            self.lr_scheduler = lr_scheduler.OneCycleLR(
                optimizer=self.optimizer, max_lr=optimizer_cfg.LR, total_steps=optimizer_cfg.total_steps)
        else:
            raise ValueError(f"Unknown Optimizer config: {optimizer_cfg}")
        
        # 3. configure lr scheduler
        lr_scheduler_config = {
            "scheduler": self.lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        if optimizer_cfg.LR_SCHEDULER == "onecycle" or optimizer_cfg.LR_SCHEDULER == "cosine_w":
            lr_scheduler_config.update({
                "interval": "step",
            })
        return {'optimizer': self.optimizer, 'lr_scheduler': lr_scheduler_config}


    ## init test meter
    @torch.no_grad()
    def on_test_epoch_start(self):
        logger.info(f'Model arch:\n{summary(self.model, verbose=0, depth=2)}')
        if self.config.TEST.withGT:
            self.test_losses_meter = {"TotalLoss": AverageMeter("Test/TotalLoss")}
            if self.config.LOSSES.SINGLE:
                self.test_losses_meter['PIXEL_LOSS'] = AverageMeter("Test/PIXEL_LOSS")
            else:
                for name in self.config.LOSSES.keys():
                    cfg_node = getattr(self.config.LOSSES, name)
                    if isinstance(cfg_node, CN) and hasattr(cfg_node, 'weight') and getattr(cfg_node, 'enable', False):
                        self.test_losses_meter[name] = AverageMeter(f"Test/{name}")
            self.test_metric_meter = {}
            for name in self.config.METRICS.keys():
                cfg_node = getattr(self.config.METRICS, name)
                if isinstance(cfg_node, CN) and hasattr(cfg_node, 'only_Y'):
                    self.test_metric_meter[name] = AverageMeter(f"Test/{name}")
            self.test_seq_metric_meter = {}

        return super().on_test_epoch_start()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # img_path = batch['gt_path']
        img_path = batch['lq_path_list'][self.nframes//2]
        
        seq_name = batch['seq_name']
        
        if isinstance(img_path, list):
            img_path = img_path[0]
        if isinstance(seq_name, list):
            seq_name = seq_name[0]
        base_img_name = f'{osp.splitext(osp.basename(img_path))[0]}.png'
        # import pdb; pdb.set_trace()
        # if batch_idx == 0:
        #     from thop import profile
        #     flops, _ = profile(self.model, inputs=(batch,))
        #     logger.debug(f'FLOPs: {flops} (= {flops/1e9} G)')
        outputs = self(batch)  # it will call LightningModule.forward()
        
        # save feature maps to validate if incurs blurred effects
        if self.config.MODEL.output_feature_vis:                            
            lq_basename = outputs['lq_basename']
            seq_name = outputs['seq_name']
            save_root = osp.join(self.config.SYSTEM.log_dir, 'test_features_visualization', f'{seq_name}_{lq_basename}')
            
            self.visualize_batch(ret=outputs, save_root=save_root)
        
        
        visualization_keys = ['pred']
        if self.config.TEST.save_gt:
            visualization_keys += ['gt']
        if self.config.TEST.save_lq:
            visualization_keys += ['lq']
        if self.config.TEST.need_illu:
            visualization_keys += ['image_illu', 'image_illu_v', 'brighten_x', 'darken_x']
        # save images
        if self.config.TEST.save_pred_img:
            """
            visualization
            default visualize ['lq', 'pred', 'gt']
            if need_illu, visualize ['image_illu', 'image_illu_v', 'brighten_x', 'darken_x']
            """
            save_dir = osp.join(self.config.SYSTEM.log_dir, 'pred_visualization')
            os.makedirs(save_dir, exist_ok=True)
            
            for k in visualization_keys:
                if k not in outputs:
                    continue
                output_k = outputs[k]
                torch.clamp(output_k, 0, 1)
                for i in range(output_k.shape[0]):
                    pred_img = tensor2img(output_k[i])
                    save_path = osp.join(save_dir, k, seq_name, base_img_name)
                    if not osp.exists(osp.dirname(save_path)):
                        os.makedirs(osp.dirname(save_path))
                    cv2.imwrite(save_path, pred_img)
                
        # with GT, calculate the loss and metric
        if self.config.TEST.withGT:
            test_losses, test_name_to_loss = self.criterion(outputs)
            test_name_to_measure = self.metrics(x=outputs['pred'], y=outputs['gt'])
            self.test_losses_meter["TotalLoss"].update(test_losses.detach().item())
            for name, loss_item in test_name_to_loss.items():
                loss_item = loss_item.detach().item()
                self.test_losses_meter[name].update(loss_item)
            for name, measure_item in test_name_to_measure:
                measure_item = measure_item.detach().item()
                self.test_metric_meter[name].update(measure_item)
            # self.log('test_loss', test_losses.detach().item(), on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.TEST.batch_size)

            # agg results in seq
            seq_name = analysis_seqname_from_filepath(img_path)
            if seq_name not in self.test_seq_metric_meter:
                self.test_seq_metric_meter[seq_name] = {}
                for name in self.config.METRICS.keys():
                    cfg_node = getattr(self.config.METRICS, name)
                    if isinstance(cfg_node, CN) and hasattr(cfg_node, 'only_Y'):
                        self.test_seq_metric_meter[seq_name][name] = []
            for name, measure_item in test_name_to_measure:
                measure_item = measure_item.detach().item()
                self.test_seq_metric_meter[seq_name][name].append(measure_item)

            if batch_idx % self.config.TEST.test_freq == 0:
                log_str = f'Test [batch: {batch_idx:05d}] [lq {img_path}]: ' + self._format_log(self.test_losses_meter, self.test_metric_meter)
                logger.info(log_str)
            return self.test_losses_meter
 

    @torch.no_grad()
    def on_test_epoch_end(self):
        if self.config.TEST.withGT:
            log_str = f'Test: ' + self._format_log(self.test_losses_meter, self.test_metric_meter)
            logger.info(log_str)
            
            for name in self.config.METRICS.keys():
                log_str = f'Sequence {name} metrics: '
                for k,v in self.test_seq_metric_meter.items():
                    seq_metric = sum(self.test_seq_metric_meter[k][name]) / len(self.test_seq_metric_meter[k][name])
                    log_str += f'{k}: {seq_metric}, '
                logger.info(log_str)

            
            for metric_name, metric_value in self.test_metric_meter.items():
                self.log(f'test_{metric_name}_epoch', metric_value.avg, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.TEST.batch_size)  # test loop will ignore: on_step=True
            
        return super().on_test_epoch_end()
