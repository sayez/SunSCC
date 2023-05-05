from typing import Optional
import pytorch_lightning as pl
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix
import torch


class IoU(ConfusionMatrix):
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        class_index: Optional[int] = None,
    ):
        super().__init__(
            num_classes,
            normalize=None,
            threshold=threshold,
            compute_on_step=True,
            dist_sync_on_step=False,
        )
        self.class_index = class_index

    def compute(self) -> torch.Tensor:
        cm = super().compute().to(torch.double)
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
        if self.class_index is None:
            return iou.mean()
        else:
            return iou[self.class_index]


class DeepsunIoU(MulticlassConfusionMatrix):
    def __init__(
        self,
        num_classes: int,
        ignore_index=None,
        threshold: float = 0.5,
        class_index: Optional[int] = None,
    ):
        # num_classes = num_classes if ignore_index is None else num_classes+1
        # print('constructor', num_classes, ignore_index)
        super().__init__(
            num_classes,
            normalize=None,
            ignore_index = ignore_index,
            threshold=threshold,
            compute_on_step=True,
            dist_sync_on_step=False,
        )
        self.class_index = class_index

    def compute(self) -> torch.Tensor:
        cm = super().compute().to(torch.double)
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
        if self.class_index is None:
            return iou.mean()
        else:
            return iou[self.class_index]

