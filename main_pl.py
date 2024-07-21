import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch import cli
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from deit_pl import DeiTModel
from datasets import build_dataset
from augment import new_data_aug_generator
from deit_args import get_deit_args_parser

def get_loaders(args):
    dataset_train, nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    return data_loader_train, data_loader_val


if __name__ == "__main__":
    # lightning args
    parser = cli.LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, 'trainer')
    parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
    parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")

    # deit args
    deit_args = get_deit_args_parser()
    parser = parser.from_argparse_args(deit_args)
    args = parser.parse_args()

    print(args)
    # trainer = pl.Trainer(args)
    # model = DeiTModel(args)
    # train_loader, val_loader = get_loaders(args)
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
