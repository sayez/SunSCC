import io
from typing import Any, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import logging
import numpy as np
from .metrics import IoU

from hydra.utils import instantiate

log = logging.getLogger(__name__)


class BaseSegment(pl.LightningModule):
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
    ):
        print('BaseSegment')
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_format = segmenter["input_format"]
        self.output_format = segmenter["output_format"]
        if hasattr(segmenter, "_target_"):
            print(segmenter)
            segmenter = instantiate(segmenter)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
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
            # task  = 'binary' if len(self.classes) == 1 else 'multiclass'
            # self.iou[f"{set}_mean"] = IoU(task, num_classes=len(self.classes) + 1)
            self.iou[f"{set}_mean"] = IoU(num_classes=len(self.classes) + 1)
            for i, name in enumerate(["bg", *self.classes]):
                self.iou[f"{set}_{name}"] = IoU(
                    # task, num_classes=len(self.classes) + 1, class_index=i
                    num_classes=len(self.classes) + 1, class_index=i
                )

    def predict(self, x):
        x = self.transfer_batch_to_device(x, self.device, 0)

        input_sample = {}
        for dtype in self.input_format:
            input_sample[dtype] = x[dtype].to(torch.float)

        return self(input_sample)

    def forward(self, x):
        # print('forward')
        # print(x)
        # for dtype in x:
        for dtype in self.input_format:
            # print(dtype)
            x[dtype] = x[dtype].unsqueeze(1).to(torch.float)
        seg = self.segmenter(x)[0]

        return seg

    # def common_step(self, batch):
    #     seg = batch["segmentation"].to(torch.long)  # .to(torch.float)
    #     # seg = batch["segmentation"].to(torch.float)  # .to(torch.float)
    #     print("seg: ", seg.shape, 'uniques : ', torch.unique(seg))
    #     # print(self.classes)
    #     # seg = F.one_hot(seg, len(self.classes)+1)
    #     seg = F.one_hot(seg, self.segmenter.n_classes)  # N,H*W -> N,H*W, C
    #     seg = seg.permute(0, 3, 1, 2)
    #     print(seg.shape)

    #     # seg[seg == 2] = 0
    #     input_sample = {}
    #     print(self.input_format)
    #     for dtype in self.input_format:
    #         input_sample[dtype] = batch[dtype].to(torch.float)

    #     # print(input_sample)
    #     seg_hat = self(input_sample)
    #     print("seg_hat: ", seg_hat.shape, 'uniques : ', torch.unique(seg_hat))        
    #     return seg, seg_hat

    def common_step(self, batch, phase='train'):
        # for k in batch:
        #     print(k, batch[k].shape)
        # print(type(batch["segmentation"]))
        if phase == 'test':
            seg = batch["segmentation"][0].to(torch.long) # for testing # .to(torch.float)
        else:
            seg = batch["segmentation"].to(torch.long)  # for training # .to(torch.float)

        input_sample = {}
        for dtype in self.input_format:
            input_sample[dtype] = batch[dtype].to(torch.float)

        #print("common_step: call model")
        seg_hat = self(input_sample)
        
        #print("common_step: ", seg.shape, seg_hat.shape)
        return seg, seg_hat

    def training_step(self, batch, batch_idx):
        bs = batch["segmentation"].shape[0]
        seg, seg_hat = self.common_step(batch, 'train')
        # print(torch.unique(seg), torch.unique(seg_hat))
        # print(self.loss.include_background)
        loss = self.loss(seg_hat, seg)
        self.log("train_loss", loss, on_epoch=True, on_step=False, logger=True, batch_size=bs)
        for name, iou in self.iou.items():
            if "train" not in name:
                continue
            iou(F.softmax(seg_hat, dim=1), seg)
            progbar = name == "train_mean"
            self.log(f"{name}iou", iou, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
            self.log(f"{name}iou_step", iou, prog_bar=progbar, batch_size=bs)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        bs = batch["segmentation"].shape[0]

        seg, seg_hat = self.common_step(batch, 'val')
        loss = self.loss(seg_hat, seg)
        self.log("val_loss", loss, batch_size=bs)
        for name, iou in self.iou.items():
            if "val" not in name:
                continue
            iou(F.softmax(seg_hat, dim=1), seg)
            progbar = name == "val_mean"
            self.log(f"{name}iou", iou, prog_bar=progbar, batch_size=bs)
        return dict(loss=loss)

    def test_step(self, batch, batch_idx):
        bs = batch["image"][0].shape[0]
        # print('test_step')
        # print(batch.keys()  )
        # print(batch["segmentation"])
        # print(batch["segmentation"].shape)
        # print(batch["segmentation"][0].shape)
        seg, seg_hat = self.common_step(batch, 'test')
        loss = self.loss(seg_hat, seg)
        self.log("test_loss", loss, batch_size=bs)
        for name, iou in self.iou.items():
            if "test" not in name:
                continue
            iou(F.softmax(seg_hat, dim=1), seg)
            progbar = name == "test_mean"
            self.log(f"{name}iou", iou, prog_bar=progbar, batch_size=bs)
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
                #"interval": "step",
                # "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "monitor": "val_loss",
            },
        )
