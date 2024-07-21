import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from .datasets import build_dataset
from .args import get_args_parser
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from augment import new_data_aug_generator
from timm.data import Mixup
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from losses import DistillationLoss
from engine import train_one_epoch, evaluate
import time
import utils
from timm.utils import accuracy, ModelEma


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.save_hyperparameters()

        print(f"Creating model: {args.model}")
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )

        criterion = LabelSmoothingCrossEntropy()
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        if mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        if args.bce_loss:
            criterion = torch.nn.BCEWithLogitsLoss()

        teacher_model = None
        if args.distillation_type != 'none':
            assert args.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {args.teacher_model}")
            teacher_model = create_model(
                args.teacher_model,
                pretrained=False,
                num_classes=args.nb_classes,
                global_pool='avg',
            )
            if args.teacher_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.teacher_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.teacher_path, map_location='cpu')
            teacher_model.load_state_dict(checkpoint['model'])
            teacher_model.eval()

        criterion = DistillationLoss(
            criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
        )

        # add
        self.model = model
        self.criterion = criterion
        self.mixup_fn = mixup_fn
        self.teacher_model = teacher_model

    def training_step(self, batch, batch_idx):
        samples, targets = batch

        if self.args.cosub:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)

        if self.args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if self.args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        outputs = self.model(samples)
        if not self.args.cosub:
            loss = self.criterion(samples, outputs, targets)
        else:
            outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
            loss = 0.25 * self.criterion(outputs[0], targets)
            loss = loss + 0.25 * self.criterion(outputs[1], targets)
            loss = loss + 0.25 * self.criterion(outputs[0], outputs[1].detach().sigmoid())
            loss = loss + 0.25 * self.criterion(outputs[1], outputs[0].detach().sigmoid())

        self.log('train/loss', loss.item())

        return loss.item()

    def validation_step(self, batch, batch_idx):
        images, target = batch
        criterion = torch.nn.CrossEntropyLoss()
        output = self.model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.log('val/loss', loss.item())
        self.log('val/acc1', acc1.item())
        self.log('val/acc5', acc5.item())

        return loss.item()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.args, self.model)
        lr_scheduler, _ = create_scheduler(self.args, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
