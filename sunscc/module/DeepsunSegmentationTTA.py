import io
from typing import Any, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import logging
import numpy as np
from .metrics import IoU, DeepsunIoU

from hydra.utils import instantiate, call

log = logging.getLogger(__name__)


import collections
from functools import partial

class DeepsunSegmentation_TTA(pl.LightningModule):
    def __init__(
        self,
        segmenter,
        lr=1e-3,
        optimizer="torch.optim.Adam",
        loss=nn.CrossEntropyLoss,
        optimizer_params=None,
        scheduler=None,
        scheduler_interval = 'epoch',
        class_weights=None,
        num_tta=5,
        transforms_tta=None
    ):
        print('DeepsunSegmentation_TTA')
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_format = segmenter["input_format"]
        self.output_format = segmenter["output_format"]
        if hasattr(segmenter, "_target_"):
            print(segmenter)
            segmenter = instantiate(segmenter)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        # hack for loss
        from omegaconf import OmegaConf

        loss = OmegaConf.create({"_target_": "sunscc.loss.CombineLosses",
                              "ignore_index": -1,
                              "sublossA":{ "_target_": "torch.nn.CrossEntropyLoss",
                                            "ignore_index": -1},
                              "sublossB":{ "_target_": "sunscc.loss.LogCosHDiceLoss",
                                            "ignore_index": -1,
                                            "softmax": True,
                                            "to_onehot_target": True,
                                            "include_background": True},
                              "ce_ratio": 0.5,
                              "include_background": True,
                           })

        if hasattr(loss, "_target_"):
            print(loss["_target_"])
            if (loss["_target_"] == "segmentation_models_pytorch.losses.dice.DiceLoss") or (
                loss["_target_"] == "sunscc.loss.GeneralizedDiceLoss" ) or (
                loss["_target_"] == "monai.losses.DiceLoss"   ) or (
                loss["_target_"] == "sunscc.loss.GDiceLoss"   
                ):
                self.loss = instantiate(loss)
            else:
                self.loss = instantiate(loss, weight=class_weights)
        else:
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.segmenter = segmenter
        self.classes = segmenter.classes
        print(self.classes)
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.iou = nn.ModuleDict()
        for set in ["train", "val", "test"]:
            task = "binary" if len(self.classes) == 1 else "multiclass"
            # self.iou[f"{set}_mean"] = IoU(num_classes=len(self.classes) + 1)
            # self.iou[f"{set}_mean"] = DeepsunIoU(num_classes=len(self.classes) + 1, ignore_index=loss.ignore_index)
            self.iou[f"{set}_mean"] = DeepsunIoU(task, num_classes=len(self.classes) + 1, ignore_index=loss.ignore_index)
            for i, name in enumerate(["bg", *self.classes]):
                # self.iou[f"{set}_{name}"] = IoU(
                #     num_classes=len(self.classes) + 1, class_index=i
                # )                
                self.iou[f"{set}_{name}"] = DeepsunIoU(
                    task, num_classes=len(self.classes) + 1, class_index=i, ignore_index=loss.ignore_index
                    # num_classes=len(self.classes) + 1, class_index=i, ignore_index=loss.ignore_index
                )


        # self.num_tta = num_tta if transforms_tta is not None else 0
        # if isinstance(transforms_tta, collections.Mapping):
        #     transforms_tta = partial(call, config=transforms_tta)
        # elif isinstance(transforms_tta, collections.Sequence):
        #     tta_transforms_init = []
        #     for transform in transforms_tta:
        #         tta_transforms_init.append(instantiate(transform))
        #     transforms_tta = tio.Compose(tta_transforms_init)
        # self.tta_transforms = transforms_tta


    def predict(self, x):
        x = self.transfer_batch_to_device(x, self.device, 0)
        return self(x)

    def forward(self, x):
        for dtype in x:
            if torch.is_tensor(x[dtype]):
                x[dtype] = x[dtype].unsqueeze(1).to(torch.float)
        seg = self.segmenter(x)[0]

        return seg


    def common_step(self, batch):
        # print(type(batch["segmentation"]))
        seg = batch["segmentation"].to(torch.long)  # .to(torch.float)
        # seg = batch["segmentation"][0].to(torch.long)  # .to(torch.float)

        input_sample = {}
        for dtype in self.input_format:
            input_sample[dtype] = batch[dtype].to(torch.float)

        seg_hat = self(input_sample)

        # print('common step',seg.shape,torch.unique(seg) ,seg_hat.shape,torch.unique(seg_hat))
        
        return seg, seg_hat

    def training_step(self, batch, batch_idx):
        seg, seg_hat = self.common_step(batch)
        loss = self.loss(seg_hat, seg)
        self.log("train_loss", loss, on_epoch=True, on_step=False, logger=True)
        for name, iou in self.iou.items():
            if "train" not in name:
                continue
            iou(F.softmax(seg_hat, dim=1), seg)
            progbar = name == "train_mean"
            self.log(f"{name}iou", iou, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{name}iou_step", iou, prog_bar=progbar)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        seg, seg_hat = self.common_step(batch)
        loss = self.loss(seg_hat, seg)
        self.log("val_loss", loss)
        for name, iou in self.iou.items():
            if "val" not in name:
                continue
            iou(F.softmax(seg_hat, dim=1), seg)
            progbar = name == "val_mean"
            self.log(f"{name}iou", iou, prog_bar=progbar)
        return dict(loss=loss)

    # def tta_test_step(self, batch):
    #     print("batch['segmentation']: ", batch['segmentation'].shape)
    #     seg = batch["segmentation"].to(torch.long)  # .to(torch.float)


    #     results = []
    #     for i in range(self.num_tta):
    #         input_sample = {}

    #         #1 TTA Transform input
    #         for dtype in self.input_format:
    #             augmented = self.tta_transforms(batch[dtype].unsqueeze(-1).to(torch.float))
    #             print(augmented)
    #             input_sample[dtype] = augmented.t1.data[None]

    #         #2 Prediction onf Transformed
    #         seg_hat = self(input_sample)


    #         #3 Add prediction to TTA object + Revert Transformation
    #         lm_temp = tio.LabelMap(tensor=torch.rand(1,1,1,1), affine=augmented.t1.affine)
    #         augmented.add_image(lm_temp, 'label')
    #         augmented.label.set_data(seg_hat)
    #         back = augmented.apply_inverse_transform(warn=True)
    #         results.append(back.label.data)

    #     result = torch.stack(results).to(torch.float)

    #     tta_result_tensor = result.mode(dim=0).values  # majority voting

    #     return seg, tta_result_tensor

    def test_step(self, batch, batch_idx):

        seg, seg_hat = self.common_step(batch)
        # print('seg_hat: ',seg_hat.shape,' seg: ', seg.shape)
        # if self.tta_transforms is None:
        #     seg, seg_hat = self.common_step(batch)
        # else:
        #     seg, seg_hat = self.tta_test_step(batch)

        loss = self.loss(seg_hat, seg)
        self.log("test_loss", loss)
        for name, iou in self.iou.items():
            if "test" not in name:
                continue
            iou(F.softmax(seg_hat, dim=1), seg)
            progbar = name == "test_mean"
            self.log(f"{name}iou", iou, prog_bar=progbar)
        return dict(loss=loss)

    def configure_optimizers(self):
        optimizer = {
            "_target_": self.optimizer_class,
            "lr": self.hparams.lr,
            **self.optimizer_params,
        }
        optimizer = instantiate(optimizer, params=self.parameters())
        if self.scheduler is not None:
            scheduler = instantiate(self.scheduler, optimizer=optimizer)
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                threshold=1e-3,
                threshold_mode="abs",
            )
        return dict(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                # "interval": "step",
                # "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "monitor": "val_loss",
            },
        )
