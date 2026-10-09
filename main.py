import os
import os.path as osp

import pickle
import torch
import torch.distributed as dist
from loguru import logger
import sys
from glob import glob
import argparse

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from configs.config import update_config

from utils.util import gen_expid, create_exp_dir, mkdir_for_config
from torchinfo import summary
import time

from models.model_lit import LitModel
from data.data_module import LitDataModule
from tqdm import tqdm


def is_main_process():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    return True

def parse_args():    
    """
    Argument parser for the main.py script.
    Returns:
        args (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser('VECNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='yaml configuration files')  # reading parameters from yaml configuration file
    parser.add_argument('--opts', help='change the config from the command-line', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def set_expdir_and_broadcast(config):
    main_proc = is_main_process()
    if main_proc:
        timestr = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_label = f"{timestr}_{config.MODEL.model_name}_{config.DATA.train.dataset}"
        expid = gen_expid(exp_label)
        config.SYSTEM.expid = expid
        config.SYSTEM.log_dir = osp.join(config.SYSTEM.root_dir, f'{exp_label}_{expid}')
        config.SYSTEM.ckpt_dir = osp.join(config.SYSTEM.root_dir, f'{exp_label}_{expid}', 'ckpt')
        mkdir_for_config(config)

        # create log
        log_str = 'test' if config.TEST.test else 'train'
        log_file = osp.join(config.SYSTEM.log_dir, f'{log_str}_{exp_label}.log')
        logger.add(log_file, level="INFO", rotation="500 MB")
        logger.info(f'python {" ".join(sys.argv)}')

        scripts = glob('*.py') + glob('configs') + glob('utils') + glob('models') + glob('data') + glob('losses')
        create_exp_dir(config.SYSTEM.log_dir, scripts_to_save=scripts)
        logger.info(f'Saved scripts to {osp.join(config.SYSTEM.log_dir, "scripts")}')
        
        logger.info(f'{args}')
        logger.info(f'\n{config}')
        # save expid, exp_label
        with open("temp_config.pkl", "wb") as f:
            pickle.dump({
                'expid': config.SYSTEM.expid,
                'log_dir': config.SYSTEM.log_dir,
                'ckpt_dir': config.SYSTEM.ckpt_dir,
                'exp_label': exp_label
            }, f)

    # master process
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # non-master
    if not main_proc:
        with open("temp_config.pkl", "rb") as f:
            temp_config = pickle.load(f)
        config.SYSTEM.expid = temp_config['expid']
        config.SYSTEM.log_dir = temp_config['log_dir']
        config.SYSTEM.ckpt_dir = temp_config['ckpt_dir']


def main(config):
    seed = config.TRAIN.seed
    pl.seed_everything(seed)
    # uncomment following line will trigger abundance of warnings
    # torch.use_deterministic_algorithms(True, warn_only=True)  # raise error if set in lightning pl.Trainer. 

    # test
    if config.TEST.test:
        if config.RESUME.ckpt_path is None:
            raise ValueError("Please specify the checkpoint path for testing")
        ckpt_path = config.RESUME.ckpt_path
        logger.info(f"Loading checkpoint from {ckpt_path}")
        if osp.isfile(ckpt_path):
            # 1. build dataloader
            test_datamodule = LitDataModule(config)
            test_datamodule.setup(stage='test')
            test_loader = test_datamodule.test_dataloader()
            # 2. load Lightning checkpoint
            lit_model = LitModel.load_from_checkpoint(ckpt_path, config=config)
            lit_model.eval()
            test_datamodule = LitDataModule(config)
            trainer = pl.Trainer(log_every_n_steps=config.TRAIN.print_freq, 
                                 inference_mode=True, 
                                 default_root_dir=config.SYSTEM.log_dir,
                                 deterministic=False, 
                                 strategy="auto")  # deterministic=True
            logger.info(f"Testing Start")
            logger.info(f'#sample={len(test_loader)}')
            total_inference_time = time.time()
            torch.cuda.synchronize()
            results = trainer.test(model=lit_model, datamodule=test_datamodule)
            
            torch.cuda.synchronize()
            total_inference_time = time.time() - total_inference_time
            if results:
                results_label = ''
                logger.info("Test results:")
                for k, v in results[0].items():
                    logger.info(f"    {k}: {v}")
                    results_label += f'{k.split("_")[1]}{v:.4f}-'
                results_label += 'test.pth'
            # save model to the the config.SYSTEM.ckpt_dir `ckpt` folder
            save_model_path = osp.join(config.SYSTEM.ckpt_dir, results_label)
            torch.save(lit_model.state_dict(), save_model_path)            
            logger.info(f"Testing Done")
            return
    
    # train
    tb_recorder = TensorBoardLogger(save_dir=config.SYSTEM.log_dir, name=config.MODEL.model_name)        
    tb_recorder.log_hyperparams({
        'batch_size': config.TRAIN.batch_size,
        'learning_rate': config.TRAIN.OPTIMIZER.LR,
        'epochs': config.TRAIN.num_epochs,
        'steps': config.TRAIN.num_steps,
        'model_name': config.MODEL.model_name,
        'optimizer': config.TRAIN.OPTIMIZER.NAME,
        'lr_scheduler': config.TRAIN.OPTIMIZER.LR_SCHEDULER,        
        'seed': config.TRAIN.seed,
        })
    
    freq_epoch_ckpt_callback = ModelCheckpoint(
        monitor='epoch',  # val_psnr, val_ssim, train_loss
        mode='max',  # min, max
        save_top_k=2,
        save_weights_only=False,
        filename=f"{config.MODEL.model_name}_"+"{epoch:06d}",
        dirpath=config.SYSTEM.ckpt_dir,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )    
    val_psnr_ckpt_callback = ModelCheckpoint(
        monitor='val_PSNR', 
        mode='max',  # min, max
        save_top_k=2,
        save_weights_only=True,  # only weights
        filename=f"{config.MODEL.model_name}_"+"{epoch:03d}_{val_PSNR:.6f}",
        dirpath=config.SYSTEM.ckpt_dir,
        every_n_epochs=1
    )
    callbacks = [freq_epoch_ckpt_callback, val_psnr_ckpt_callback]
    trainer = pl.Trainer(logger=tb_recorder, 
                         max_epochs=config.TRAIN.num_epochs, 
                         max_steps=config.TRAIN.num_steps,
                         callbacks=callbacks,
                         log_every_n_steps=config.TRAIN.print_freq,
                         check_val_every_n_epoch=config.TRAIN.val_freq_epoch, 
                         default_root_dir=config.SYSTEM.log_dir, 
                         deterministic=False,
                         strategy="auto")  # deterministic=True
    
    # create model
    lit_model = LitModel(config)
    data_module = LitDataModule(config)

    # Start training (with or without checkpoint)
    if config.RESUME.resume:
        # resume training
        ckpt_path = config.RESUME.ckpt_path
        if osp.isfile(ckpt_path):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            
            trainer.fit(model=lit_model, datamodule=data_module, ckpt_path=ckpt_path)
            logger.info(f"Resumed training from checkpoint")
        else:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    else:
        # train from scratch
        trainer.fit(model=lit_model, datamodule=data_module)
    

if __name__ == '__main__':
    args = parse_args()
    config = update_config(args.config, args.opts)
    
    config.defrost()
    # defrost config to set logdir/ckptdir/tbdir
    gpu_count = torch.cuda.device_count()
    logger.info(f'Using {gpu_count} GPU(s).')
    if gpu_count > 1:
        set_expdir_and_broadcast(config)
    else:
        timestr = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_label = f"{timestr}_{config.MODEL.model_name}_{config.DATA.train.dataset}"
        expid = gen_expid(exp_label)
        config.SYSTEM.expid = expid
        config.SYSTEM.log_dir = osp.join(config.SYSTEM.root_dir, f'{exp_label}_{expid}')
        config.SYSTEM.ckpt_dir = osp.join(config.SYSTEM.root_dir, f'{exp_label}_{expid}', 'ckpt')
        mkdir_for_config(config)

        # create log
        log_str = 'test' if config.TEST.test else 'train'
        log_file = osp.join(config.SYSTEM.log_dir, f'{log_str}_{exp_label}.log')
        logger.add(log_file, level="INFO", rotation="500 MB")
        logger.info(f'python {" ".join(sys.argv)}')

        scripts = glob('*.py') + glob('configs') + glob('utils') + glob('models') + glob('data') + glob('losses')
        create_exp_dir(config.SYSTEM.log_dir, scripts_to_save=scripts)
        logger.info(f'Saved scripts to {osp.join(config.SYSTEM.log_dir, "scripts")}')
        

    if config.SYSTEM.debug:
        logger.info(f'Debug mode, change frequency...')
        config.TRAIN.print_freq = 10
        config.TRAIN.save_ckpt_freq = 1
        config.VALIDATE.val_freq = 1  # unit is epoch
        config.TRAIN.num_epochs = 2
        config.TRAIN.num_steps = 500
        config.SYSTEM.root_dir = 'exp_debug'
        
    logger.info(f'{args}')
    logger.info(f'\n{config}')
    config.freeze()
    main(config)
    logger.info(f'Finish.')
    