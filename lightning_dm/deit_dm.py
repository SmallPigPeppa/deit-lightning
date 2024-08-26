import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from datasets import build_dataset
from augment import new_data_aug_generator
from lightning_dm.sampler import RASampler
class CustomDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        self.dataset_train, self.args.nb_classes = build_dataset(is_train=True, args=self.args)
        self.dataset_val, _ = build_dataset(is_train=False, args=self.args)

        if self.args.ThreeAugment:
            self.dataset_train.transform = new_data_aug_generator(self.args)

        if self.args.repeated_aug:
            self.sampler_train = RASampler(self.dataset_train, shuffle=True, repetitions=self.args.ra_reps)
        else:
            self.sampler_train = torch.utils.data.distributed.DistributedSampler(self.dataset_train)
        self.sampler_val = torch.utils.data.distributed.DistributedSampler(self.dataset_val, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            sampler=self.sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            sampler=self.sampler_val,
            batch_size=int(1.5 * self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=False
        )
