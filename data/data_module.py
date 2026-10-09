import lightning.pytorch as pl
from data import build_dataset
from torch.utils.data import DataLoader

class LitDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Add state tracking for resumable training
        self._current_epoch = 0
        self._current_step = 0

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = build_dataset(self.config, mode='train')
            self.val_dataset = build_dataset(self.config, mode='validate')
        elif stage == 'test':
            self.test_dataset = build_dataset(self.config, mode=stage)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.TRAIN.batch_size,
            num_workers=self.config.TRAIN.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.VALIDATE.batch_size,
            num_workers=self.config.VALIDATE.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.TEST.batch_size,
            num_workers=self.config.TEST.num_workers,
            pin_memory=True
        )
    
    def state_dict(self):
        """Save the state of the datamodule for resumable training"""
        return {
            'current_epoch': self._current_epoch,
            'current_step': self._current_step,
        }
    
    def load_state_dict(self, state_dict):
        """Load the state of the datamodule for resumable training"""
        self._current_epoch = state_dict.get('current_epoch', 0)
        self._current_step = state_dict.get('current_step', 0)