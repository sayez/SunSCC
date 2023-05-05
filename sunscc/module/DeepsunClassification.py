from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler

import logging
import numpy as np

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics

from hydra.utils import instantiate

import time

class McIntoshClassifier_Superclasses(pl.LightningModule):
    def __init__(
                self,
                classifier,
                lr=1e-3, 
                optimizer='torch.optim.Adam', 
                loss=nn.CrossEntropyLoss,
                optimizer_params=None,
                scheduler=None,
                class1_weights = None,
                class2_weights = None,
                class3_weights = None,
                scheduler_interval = 'epoch',
                batch_size=16,
                transfer=True, 
                tune_fc_only=True
                ):
        super().__init__()

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.scheduler_interval = scheduler_interval

        # self.input_format = classifier["input_format"]
        self.visual_input_format = classifier["input_format"]['visual']
        self.numeric_input_format = classifier["input_format"]['numeric']
        self.input_format = self.visual_input_format + self.numeric_input_format

        self.output_format = classifier["output_format"]

        print(classifier)

        if hasattr(classifier, "_target_"):
            classifier = instantiate(classifier)

        if class1_weights is not None:
            class1_weights = torch.tensor(class1_weights, dtype=torch.float)
        if class2_weights is not None:
            class2_weights = torch.tensor(class2_weights, dtype=torch.float)
        if class3_weights is not None:
            class3_weights = torch.tensor(class3_weights, dtype=torch.float)

        # print(classifier)

        self.classifier = classifier
        self.classes = self.classifier.classes

        print(self.classes)

        self.num_classes = len(self.classes)

        if loss['_target_'] == "torch.nn.CrossEntropyLoss":
            self.loss1 = instantiate(loss,weight=class1_weights)
            self.loss2 = instantiate(loss,weight=class2_weights)
            self.loss3 = instantiate(loss,weight=class3_weights)
        else:
            self.loss1 = instantiate(loss)
            self.loss2 = instantiate(loss)
            self.loss3 = instantiate(loss)

        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler = scheduler

    def predict(self, X):
        X = self.transfer_batch_to_device(X, self.device, 0)
        return self(X)

    def forward(self, X):
        # print('classifier_forward')
        # return self.classifier(X)
        for dtype in X:
            if not torch.is_tensor(X[dtype]):
                continue
            X[dtype] =  X[dtype].to(torch.float)
            if len(X[dtype].shape) == 3:
                X[dtype] = X[dtype].unsqueeze(1).to(torch.float)
            if len(X[dtype].shape) == 4:
                X[dtype] = X[dtype].to(torch.float)
        classif1, classif2,classif3 = self.classifier(X)
        return classif1, classif2,classif3


    def common_step(self, batch):
        # print('common_step')
        classif1 = torch.squeeze(batch["class1"],1)
        classif2 = torch.squeeze(batch["class2"],1)
        classif3 = torch.squeeze(batch["class3"],1)

        input_sample = {}
        for dtype in self.input_format:
            input_sample[dtype] = batch[dtype].to(torch.float)

        # print('common_step: compute_hat')
        classif1_hat, classif2_hat, classif3_hat = self(input_sample)

        # print(f'common_step results: {classif.shape} (GT), {classif_hat.shape} (pred)')

        return classif1, classif1_hat, classif2, classif2_hat, classif3, classif3_hat,
    
    def training_step(self, batch, batch_idx):
        st = time.time()
        # classif, classif_hat = self.common_step(batch)
        c1, c1_hat, c2, c2_hat, c3, c3_hat = self.common_step(batch)
        et = time.time()
        
        ############################
        st = time.time()
        loss1 = self.loss1(c1_hat, c1)
        loss2 = self.loss2(c2_hat, c2)
        loss3 = self.loss3(c3_hat, c3)

        acc_fct1 = torchmetrics.Accuracy().to(c1_hat.device)
        acc_fct2 = torchmetrics.Accuracy().to(c2_hat.device)
        acc_fct3 = torchmetrics.Accuracy().to(c3_hat.device)

        pred1 = c1_hat.softmax(dim=-1)
        acc1 = acc_fct1(pred1,c1)
        pred2 = c2_hat.softmax(dim=-1)
        acc2 = acc_fct2(pred2,c2)
        pred3 = c3_hat.softmax(dim=-1)
        acc3 = acc_fct3(pred3,c3)

        self.log("train_loss1", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        self.log("train_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        
        self.log("train_loss2", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        self.log("train_acc2", acc2, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        
        self.log("train_loss3", loss3, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        self.log("train_acc3", acc3, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        
        et = time.time()
        # print(f'forward time: {et-st} seconds')
        
        return dict(loss=loss1+loss2+loss3)

    
    def validation_step(self, batch, batch_idx):
        st = time.time()
        # classif, classif_hat = self.common_step(batch)
        c1, c1_hat, c2, c2_hat, c3, c3_hat = self.common_step(batch)
        et = time.time()
        # print(f'forward time: {et-st} seconds')

        ############################
        st = time.time()
        loss1 = self.loss1(c1_hat, c1)
        loss2 = self.loss2(c2_hat, c2)
        loss3 = self.loss3(c3_hat, c3)

        acc_fct1 = torchmetrics.Accuracy().to(c1_hat.device)
        acc_fct2 = torchmetrics.Accuracy().to(c2_hat.device)
        acc_fct3 = torchmetrics.Accuracy().to(c3_hat.device)

        pred1 = c1_hat.softmax(dim=-1)
        acc1 = acc_fct1(pred1,c1)
        pred2 = c2_hat.softmax(dim=-1)
        acc2 = acc_fct2(pred2,c2)
        pred3 = c3_hat.softmax(dim=-1)
        acc3 = acc_fct3(pred3,c3)

        self.log("val_loss1", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        self.log("val_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        
        self.log("val_loss2", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        self.log("val_acc2", acc2, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        
        self.log("val_loss3", loss3, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        self.log("val_acc3", acc3, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])

        global_acc = float(0)
        self.log("global_acc", global_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        
        et = time.time()
        
        return dict(loss=loss1+loss2+loss3)

    def test_step(self, batch, batch_idx):
        st = time.time()
        # classif, classif_hat = self.common_step(batch)
        c1, c1_hat, c2, c2_hat, c3, c3_hat = self.common_step(batch)
        
        acc_fct1 = torchmetrics.Accuracy().to(c1_hat.device)
        acc_fct2 = torchmetrics.Accuracy().to(c2_hat.device)
        acc_fct3 = torchmetrics.Accuracy().to(c3_hat.device)

        pred1 = c1_hat.softmax(dim=-1)
        acc1 = acc_fct1(pred1,c1)
        pred2 = c2_hat.softmax(dim=-1)
        acc2 = acc_fct2(pred2,c2)
        pred3 = c3_hat.softmax(dim=-1)
        acc3 = acc_fct3(pred3,c3)

        self.log("test_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])     
        self.log("test_acc2", acc2, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])   
        self.log("test_acc3", acc3, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])

        global_acc = float(0)
        self.log("global_test_acc", global_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["class1"].shape[0])
        
        et = time.time()


    def test_step_OLD(self, batch, batch_idx):
        # print()
        classif, classif_hat = self.common_step(batch)

        if self.num_classes == 2:
            classif = F.one_hot(classif, num_classes=2).float()
        
        loss = self.loss(classif_hat, classif)

        if self.num_classes == 2:
            acc = (torch.argmax(classif,1) == torch.argmax(classif_hat,1)) \
                    .type(torch.FloatTensor).mean()
        else:
            # print(classif,torch.argmax(classif_hat,1))

            batch_acc = torchmetrics.Accuracy().to(classif_hat.device)
            # print(classif_hat.shape, classif_hat.device)
            # print(classif.shape, classif.device)
            pred = classif_hat.softmax(dim=-1)
            # print(pred.device)

            acc = batch_acc(pred,classif)
            # print(f'Accuracy on batch: {acc}')
        
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
                

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
                "frequency": 1,
                "reduce_on_plateau": True,
                "monitor": "val_loss",
            },
        )



class McIntoshClassifier_SuperclassesOLD(pl.LightningModule):
    def __init__(
                self,
                classifier,
                lr=1e-3, 
                optimizer='torch.optim.Adam', 
                loss=nn.CrossEntropyLoss,
                optimizer_params=None,
                scheduler=None,
                class1_weights = None,
                class2_weights = None,
                class3_weights = None,
                batch_size=16,
                transfer=True, 
                tune_fc_only=True
                ):
        super().__init__()

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.input_format = classifier["input_format"]
        self.output_format = classifier["output_format"]

        # print(classifier)

        if hasattr(classifier, "_target_"):
            classifier = instantiate(classifier)

        if class1_weights is not None:
            class1_weights = torch.tensor(class1_weights, dtype=torch.float)
        if class2_weights is not None:
            class2_weights = torch.tensor(class2_weights, dtype=torch.float)
        if class3_weights is not None:
            class3_weights = torch.tensor(class3_weights, dtype=torch.float)

        # print(classifier)

        self.classifier = classifier
        self.classes = self.classifier.classes

        print(self.classes)

        self.num_classes = len(self.classes)

        if loss['_target_'] == "torch.nn.CrossEntropyLoss":
            self.loss1 = instantiate(loss,weight=class1_weights)
            self.loss2 = instantiate(loss,weight=class2_weights)
            self.loss3 = instantiate(loss,weight=class3_weights)
        else:
            self.loss1 = instantiate(loss)
            self.loss2 = instantiate(loss)
            self.loss3 = instantiate(loss)

        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler = scheduler

    def predict(self, X):
        X = self.transfer_batch_to_device(X, self.device, 0)
        return self(X)

    def forward(self, X):
        # print('classifier_forward')
        # return self.classifier(X)
        for dtype in X:
            if not torch.is_tensor(X[dtype]):
                continue
            X[dtype] =  X[dtype].to(torch.float)
            if len(X[dtype].shape) == 3:
                X[dtype] = X[dtype].unsqueeze(1).to(torch.float)
            if len(X[dtype].shape) == 4:
                X[dtype] = X[dtype].to(torch.float)
        classif1, classif2,classif3 = self.classifier(X)
        return classif1, classif2,classif3


    def common_step(self, batch):
        # print('common_step')
        classif1 = torch.squeeze(batch["class1"],1)
        classif2 = torch.squeeze(batch["class2"],1)
        classif3 = torch.squeeze(batch["class3"],1)

        input_sample = {}
        for dtype in self.input_format:
            input_sample[dtype] = batch[dtype].to(torch.float)

        # print('common_step: compute_hat')
        classif1_hat, classif2_hat, classif3_hat = self(input_sample)

        # print(f'common_step results: {classif.shape} (GT), {classif_hat.shape} (pred)')

        return classif1, classif1_hat, classif2, classif2_hat, classif3, classif3_hat,
    
    def training_step(self, batch, batch_idx):
        st = time.time()
        # classif, classif_hat = self.common_step(batch)
        c1, c1_hat, c2, c2_hat, c3, c3_hat = self.common_step(batch)
        et = time.time()
        # print(f'forward time: {et-st} seconds')

        if self.num_classes == 2:
            c1 = F.one_hot(c1, num_classes=2).float()
        
        ############################
        st = time.time()
        loss1 = self.loss1(c1_hat, c1)
        loss2 = self.loss2(c2_hat, c2)
        loss3 = self.loss3(c3_hat, c3)

        acc_fct1 = torchmetrics.Accuracy().to(c1_hat.device)
        acc_fct2 = torchmetrics.Accuracy().to(c2_hat.device)
        acc_fct3 = torchmetrics.Accuracy().to(c3_hat.device)

        pred1 = c1_hat.softmax(dim=-1)
        acc1 = acc_fct1(pred1,c1)
        pred2 = c2_hat.softmax(dim=-1)
        acc2 = acc_fct2(pred2,c2)
        pred3 = c3_hat.softmax(dim=-1)
        acc3 = acc_fct3(pred3,c3)

        self.log("train_loss1", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("train_loss2", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc2", acc2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("train_loss3", loss3, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc3", acc3, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        et = time.time()
        
        return dict(loss=loss1+loss2+loss3)

    
    def validation_step(self, batch, batch_idx):
        st = time.time()
        # classif, classif_hat = self.common_step(batch)
        c1, c1_hat, c2, c2_hat, c3, c3_hat = self.common_step(batch)
        et = time.time()
        # print(f'forward time: {et-st} seconds')

        if self.num_classes == 2:
            c1 = F.one_hot(c1, num_classes=2).float()
        
        ############################
        st = time.time()
        loss1 = self.loss1(c1_hat, c1)
        loss2 = self.loss2(c2_hat, c2)
        loss3 = self.loss3(c3_hat, c3)

        acc_fct1 = torchmetrics.Accuracy().to(c1_hat.device)
        acc_fct2 = torchmetrics.Accuracy().to(c2_hat.device)
        acc_fct3 = torchmetrics.Accuracy().to(c3_hat.device)

        pred1 = c1_hat.softmax(dim=-1)
        acc1 = acc_fct1(pred1,c1)
        pred2 = c2_hat.softmax(dim=-1)
        acc2 = acc_fct2(pred2,c2)
        pred3 = c3_hat.softmax(dim=-1)
        acc3 = acc_fct3(pred3,c3)

        self.log("val_loss1", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("val_loss2", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc2", acc2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("val_loss3", loss3, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc3", acc3, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        global_acc = float(0)
        self.log("global_acc", global_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        et = time.time()
        
        return dict(loss=loss1+loss2+loss3)

    def test_step(self, batch, batch_idx):
        st = time.time()
        # classif, classif_hat = self.common_step(batch)
        c1, c1_hat, c2, c2_hat, c3, c3_hat = self.common_step(batch)

        if self.num_classes == 2:
            c1 = F.one_hot(c1, num_classes=2).float()
        
        acc_fct1 = torchmetrics.Accuracy().to(c1_hat.device)
        acc_fct2 = torchmetrics.Accuracy().to(c2_hat.device)
        acc_fct3 = torchmetrics.Accuracy().to(c3_hat.device)

        pred1 = c1_hat.softmax(dim=-1)
        acc1 = acc_fct1(pred1,c1)
        pred2 = c2_hat.softmax(dim=-1)
        acc2 = acc_fct2(pred2,c2)
        pred3 = c3_hat.softmax(dim=-1)
        acc3 = acc_fct3(pred3,c3)

        self.log("test_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True)     
        self.log("test_acc2", acc2, on_step=True, on_epoch=True, prog_bar=True, logger=True)   
        self.log("test_acc3", acc3, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        global_acc = float(0)
        self.log("global_test_acc", global_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        et = time.time()


    def test_step_OLD(self, batch, batch_idx):
        # print()
        classif, classif_hat = self.common_step(batch)

        if self.num_classes == 2:
            classif = F.one_hot(classif, num_classes=2).float()
        
        loss = self.loss(classif_hat, classif)

        if self.num_classes == 2:
            acc = (torch.argmax(classif,1) == torch.argmax(classif_hat,1)) \
                    .type(torch.FloatTensor).mean()
        else:
            # print(classif,torch.argmax(classif_hat,1))

            batch_acc = torchmetrics.Accuracy().to(classif_hat.device)
            # print(classif_hat.shape, classif_hat.device)
            # print(classif.shape, classif.device)
            pred = classif_hat.softmax(dim=-1)
            # print(pred.device)

            acc = batch_acc(pred,classif)
        
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
                

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
                # "interval": "step",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "monitor": "val_loss",
            },
        )
     