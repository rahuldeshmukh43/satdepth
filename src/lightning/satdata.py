import os
import pytorch_lightning as pl
from torch import distributed as dist
from loguru import logger

from satdepth.src.datasets.satdepth import SatDepthLoader

class SatDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.rot_aug = args.rot_aug

    def setup(self, stage:str=None):
        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set world_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = None #ideally this is where you would instantiate the dataset
            self.val_dataset = None

            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else: # stage == "test"
            NotImplementedError("DataModule not implemented for testing stage")

    def train_dataloader(self):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.rot_aug:
            dataloader = SatDepthLoader(self.args, "train", rotation_augmentation=True).load_data()
        else:
            dataloader = SatDepthLoader(self.args, "train", rotation_augmentation=False).load_data()
        return dataloader

    def val_dataloader(self):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        dataloader = SatDepthLoader(self.args, "val", rotation_augmentation=False).load_data()
        return dataloader