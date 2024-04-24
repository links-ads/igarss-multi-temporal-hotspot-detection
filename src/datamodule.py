from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mthd.dataset import (
    BANDS_GROUPS_IDX,
    NUM_LC_CLASSES,
    CustomSampler,
    MeteosatDataset,
)


class MeteosatDataModule(LightningDataModule):

    def __init__(self,
                 train_catalog_file,
                 val_catalog_file,
                 test_catalog_file,
                 mask_params,
                 batch_size: int = 32):
        super().__init__()
        self.train_catalog_file = train_catalog_file
        self.val_catalog_file = val_catalog_file
        self.test_catalog_file = test_catalog_file
        self.batch_size = batch_size
        self.mask_params = mask_params
        self.num_lc_classes = NUM_LC_CLASSES
        self.bands_groups = BANDS_GROUPS_IDX

    def setup(self, stage=None):
        self.train_dataset = MeteosatDataset(self.train_catalog_file,
                                             self.mask_params,
                                             max_timesteps=96)
        self.val_dataset = MeteosatDataset(self.val_catalog_file,
                                           self.mask_params,
                                           max_timesteps=96)
        self.test_dataset = MeteosatDataset(self.test_catalog_file,
                                            self.mask_params,
                                            max_timesteps=96)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          sampler=CustomSampler(self.train_dataset))

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          sampler=CustomSampler(self.val_dataset))

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          sampler=CustomSampler(self.test_dataset))
