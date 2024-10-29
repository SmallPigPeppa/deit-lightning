import torch
from timm.data import Mixup
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from losses import DistillationLoss
from lightning import LightningModule
from models.vision_transformer import vit_small_patch16_224


class DeiTModel(LightningModule):
    def __init__(
            self,
            config,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.args = config
        self.save_hyperparameters()

        print(f"Creating model: {config.model}")
        # model = create_model(
        #     config.model,
        #     pretrained=True,
        #     num_classes=config.nb_classes,
        #     drop_rate=config.drop,
        #     drop_path_rate=config.drop_path,
        #     drop_block_rate=None,
        #     img_size=config.input_size
        # )
        model = vit_small_patch16_224(pretrained=True)

        criterion = LabelSmoothingCrossEntropy()
        mixup_fn = None
        mixup_active = config.mixup > 0 or config.cutmix > 0. or config.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, cutmix_minmax=config.cutmix_minmax,
                prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
                label_smoothing=config.smoothing, num_classes=config.nb_classes)

        if mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        if config.bce_loss:
            criterion = torch.nn.BCEWithLogitsLoss()

        teacher_model = None
        if config.distillation_type != 'none':
            assert config.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {config.teacher_model}")
            teacher_model = create_model(
                config.teacher_model,
                pretrained=False,
                num_classes=config.nb_classes,
                global_pool='avg',
            )
            if config.teacher_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    config.teacher_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(config.teacher_path, map_location='cpu')
            teacher_model.load_state_dict(checkpoint['model'])
            teacher_model.eval()

        criterion = DistillationLoss(
            criterion, teacher_model, config.distillation_type, config.distillation_alpha, config.distillation_tau
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

        self.log('train/loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        criterion = torch.nn.CrossEntropyLoss()
        output = self.model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.log('val/loss', loss, sync_dist=True)
        self.log('val/acc1', acc1, sync_dist=True)
        self.log('val/acc5', acc5, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, target = batch
        criterion = torch.nn.CrossEntropyLoss()
        output = self.model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.log('test/loss', loss, sync_dist=True)
        self.log('test/acc1', acc1, sync_dist=True)
        self.log('test/acc5', acc5, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.args, self.model)
        lr_scheduler, _ = create_scheduler(self.args, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
