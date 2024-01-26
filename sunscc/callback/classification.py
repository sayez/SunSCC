import logging
from typing import Any, Optional
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
from torchmetrics.utilities.data import to_categorical
from sunscc.plot import cm
import numpy as np
from tqdm.auto import tqdm
import os
import skimage.io as io
from itkwidgets import view as view3d
import ipywidgets as widgets
from IPython.display import display
import matplotlib
import json

import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

log = logging.getLogger(__name__)


def display_first_channel_batch2(batch, image_key, predictions, gt, mapper):
    image = batch[image_key].cpu()
    bs = image.shape[0]
    
    show_classes = ['0','1']
    # show_classes = ['A','B']
    print(mapper)
    log.debug(image.ndim)

    if image.ndim == 4:
        # print('2')
        for i in range(bs):
            img = 3500* image[i, 0, :, :].cpu().numpy()
            pred = str(predictions[i].cpu().numpy())
            g = str(gt[i].cpu().numpy()[0])
            
            if (pred in show_classes) and (g in show_classes) and (g != pred):
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(3, 3),  dpi=200)

                normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)
                ax.imshow(img, cmap='gray', norm = normalize)
                
                ax.set_title(f"GT: {mapper[g]}, Pred: {mapper[pred]}")
                plt.show()
                plt.close(fig)


def display_first_channel_batch(batch, image_key, predictions, gt, mapper):
    image = batch[image_key].cpu()
    bs = image.shape[0]
    print(bs)

    log.debug(image.ndim)
    # print(mapper)
    # print('1')
    if image.ndim == 4:
        # print('2')
        fig, axs = plt.subplots(ncols=bs, figsize=(bs * 5, 10), squeeze=False, dpi=200)
        # fig, axs = plt.subplots(ncols=bs, figsize=(len(include_index) * 5, 10), squeeze=False, dpi=200)
        for i, ax in enumerate(axs[0]):
            img = 3500* image[i, 0, :, :].cpu().numpy()
            pred = str(predictions[i].cpu().numpy())
            g = str(gt[i].cpu().numpy()[0])
            
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)
            ax.imshow(img, cmap='gray', norm = normalize)
            
            ax.set_title(f"GT: {mapper[g]}, Pred: {mapper[pred]}")
        plt.show()
        plt.close(fig)


class ShowClassificationPredictionsCallback(pl.Callback):
    def __init__(self, output_dir) -> None:
        self.output_dir = Path(output_dir)
        
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)


    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
       
        used_classes = trainer.test_dataloaders[0].dataset.classes_mapper
        used_classes_lst = used_classes.items()
        inverted_mapper = {str(v): k  for k,v in used_classes_lst }
        
        classification = batch['class']

        classification_hat = pl_module(batch)
        
        classification_hat = F.softmax(classification_hat, dim=1)
        classification_hat = torch.argmax(classification_hat,dim=1)
        
        display_first_channel_batch2(batch, 'image', classification_hat, classification, inverted_mapper)
        

class ClassificationConfusionMatrixCallback(pl.Callback):
    def __init__(self, normalize=None) -> None:
        assert (normalize == None ) or (normalize =='true') or (normalize == 'pred') or (normalize == 'all')

        self.normalize = normalize


    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        
        self.total_predictions = []
        self.total_gt = []


    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
       
        used_classes = trainer.test_dataloaders[0].dataset.classes_mapper
        used_classes_lst = used_classes.items()
        inverted_mapper = {str(v): k  for k,v in used_classes_lst }
       
        classification = batch['class']

        classification_hat = pl_module(batch)
       
        classification_hat = F.softmax(classification_hat, dim=1)
        classification_hat = torch.argmax(classification_hat,dim=1)
       
        self.total_gt.append(classification.cpu().numpy())
        self.total_predictions.append(np.expand_dims(classification_hat.cpu().numpy(),axis=1))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        used_classes = trainer.test_dataloaders[0].dataset.classes_mapper
        class_names  = list(used_classes.keys())

        y_true = np.concatenate(self.total_gt, axis=0).ravel()
        y_pred = np.concatenate(self.total_predictions, axis=0).ravel()

        conf_mat = confusion_matrix(y_true,y_pred, normalize=self.normalize )

        # print(conf_mat)

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=200)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                                        display_labels=class_names,
                                        )
                                    
        disp.plot(ax=ax)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        fig.show()


####################  McIntosh

def display_ImgInput_batchMcIntosh(batch, image_key, mask_key, gt, mappers):
    image = batch[image_key].cpu()

    if image.ndim == 3:
        image = image.unsqueeze(1)
         
    mask = batch[mask_key].cpu()

    bs = image.shape[0]
    
    log.debug(image.ndim)
    
    if image.ndim == 4:
    
        fig, ax = plt.subplots(nrows=image.shape[1]+ mask.shape[1]-1 ,ncols=bs, 
                                figsize=(3*bs, 3*image.shape[1]+mask.shape[1] ),  dpi=200)

        for i in range(bs):
            img = image[i, 0, :, :].cpu().numpy()
    
            img_pen = mask[i, 1, :, :].cpu().numpy()
            img_um = mask[i, 2, :, :].cpu().numpy()

            g_C1 = str(gt[0][i].cpu().numpy()[0])
            g_C2 = str(gt[1][i].cpu().numpy()[0])
            g_C3 = str(gt[2][i].cpu().numpy()[0])

            strMcIntosh = mappers[0][g_C1] + mappers[1][g_C2] + mappers[2][g_C3]

            ax[0,i].set_title(f"GT: {strMcIntosh}")
        
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)
            ax[0,i].imshow(img, cmap='gray', norm = normalize)
            ax[1,i].imshow(img_pen, cmap='gray', interpolation=None)
            ax[2,i].imshow(img_um, cmap='gray', interpolation=None)
        
        plt.show()
        plt.close(fig)

def display_predictions_batchMcIntosh(batch, image_key, predictions, mappers):
    image = batch[image_key].cpu()

    bs = image.shape[0]
    
    log.debug(image.ndim)
   
    if image.ndim == 4:
   
        fig, ax = plt.subplots(nrows=1 ,ncols=bs, figsize=(3*bs,3),  dpi=200)

        for i in range(bs):
   
            img = image[i, 0, :, :].cpu().numpy()

            pred_C1 = str(predictions[0][i].cpu().numpy())
            pred_C2 = str(predictions[1][i].cpu().numpy())
            pred_C3 = str(predictions[2][i].cpu().numpy())

            strMcIntosh = mappers[0][pred_C1] + mappers[1][pred_C2] + mappers[2][pred_C3]

            ax[i].set_title(f"Pred: {strMcIntosh}")
        
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)
            ax[i].imshow(img, cmap='gray', norm = normalize)
        
        plt.show()
        plt.close(fig)
    
class ShowMcIntoshClassificationInputOutputsCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        return


    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, unused: int = 0) -> None:
        if trainer.sanity_checking:
            return

        used_c1 = trainer.train_dataloader.dataset.datasets.c1_mapper.items()
        used_c2 = trainer.train_dataloader.dataset.datasets.c2_mapper.items()
        used_c3 = trainer.train_dataloader.dataset.datasets.c3_mapper.items()
        inverted_mappers =   ({str(v): k  for k,v in used_c1 } ,
                                {str(v): k  for k,v in used_c2 },
                                {str(v): k  for k,v in used_c3 })
        
        c1,c2,c3 = batch['class1'], batch['class2'], batch['class3']

        McIntosh = c1,c2,c3

        display_ImgInput_batchMcIntosh(batch,'image', "mask_one_hot", McIntosh, inverted_mappers)


    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, unused: int = 0) -> None:
        if trainer.sanity_checking:
            return


        used_c1 = trainer.train_dataloader.dataset.datasets.c1_mapper.items()
        used_c2 = trainer.train_dataloader.dataset.datasets.c2_mapper.items()
        used_c3 = trainer.train_dataloader.dataset.datasets.c3_mapper.items()
        inverted_mappers =   ({str(v): k  for k,v in used_c1 } ,
                                {str(v): k  for k,v in used_c2 },
                                {str(v): k  for k,v in used_c3 })

        c1_hat,c2_hat,c3_hat = pl_module(batch)
        
        c1_hat = F.softmax(c1_hat, dim=1)
        c2_hat = F.softmax(c2_hat, dim=1)
        c3_hat = F.softmax(c3_hat, dim=1)
        
        c1_hat = torch.argmax(c1_hat,dim=1)
        c2_hat = torch.argmax(c2_hat,dim=1)
        c3_hat = torch.argmax(c3_hat,dim=1)
        
        McIntosh_hat = c1_hat,c2_hat,c3_hat

        display_predictions_batchMcIntosh(batch,'image', McIntosh_hat, inverted_mappers)


class ShowMcIntoshClassificationPredictionsCallback(pl.Callback):
    def __init__(self, output_dir) -> None:
        self.output_dir = Path(output_dir)
        
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
       

        used_c1 = trainer.test_dataloaders[0].dataset.c1_mapper.items()
        used_c2 = trainer.test_dataloaders[0].dataset.c2_mapper.items()
        used_c3 = trainer.test_dataloaders[0].dataset.c3_mapper.items()
        inverted_mappers =   ({str(v): k  for k,v in used_c1 } ,
                                {str(v): k  for k,v in used_c2 },
                                {str(v): k  for k,v in used_c3 })
        
        c1,c2,c3 = batch['class1'], batch['class2'], batch['class3']

        c1_hat,c2_hat,c3_hat = pl_module(batch)
        
        c1_hat = F.softmax(c1_hat, dim=1)
        c2_hat = F.softmax(c2_hat, dim=1)
        c3_hat = F.softmax(c3_hat, dim=1)
        # print(classification, 'vs', classification_hat)
        c1_hat = torch.argmax(c1_hat,dim=1)
        c2_hat = torch.argmax(c2_hat,dim=1)
        c3_hat = torch.argmax(c3_hat,dim=1)
        # print(classification, 'vs', classification_hat)

        McIntosh = c1,c2,c3
        McIntosh_hat = c1_hat,c2_hat,c3_hat

        display_first_channel_batchMcIntosh(batch, 'image', McIntosh_hat, McIntosh, inverted_mappers)

class McIntoshAngularCorrectnessCallback(pl.Callback):
    def __init__(self, normalize=None) -> None:
        assert (normalize == None ) or (normalize =='true') or (normalize == 'pred') or (normalize == 'all')

        self.normalize = normalize
    
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.total_angular_excentricity = []

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
        
    
        self.total_angular_excentricity.append(batch['angular_excentricity'].cpu().numpy())

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return

        self.total_angular_excentricity = np.concatenate(self.total_angular_excentricity, axis=0)

        # show an histogram of the angular excentricity
        fig, ax = plt.subplots()
        ax.hist(self.total_angular_excentricity, bins=50)
        ax.set_title('Angular excentricity')
        ax.set_xlabel('Angular excentricity')
        ax.set_ylabel('Number of samples')
        plt.show()



class McIntoshClassificationConfusionMatrixCallback(pl.Callback):
    def __init__(self, normalize=None, output_dir=None) -> None:
        assert (normalize == None ) or (normalize =='true') or (normalize == 'pred') or (normalize == 'all')

        self.normalize = normalize
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            if not self.output_dir.is_dir():
                self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        print('output_dir', self.output_dir)



    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        
        self.total_predictions1 = []
        self.total_gt1 = []
        self.total_predictions2 = []
        self.total_gt2 = []
        self.total_predictions3 = []
        self.total_gt3 = []

        self.total_angular_excentricity = []


    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
       
        used_c1 = trainer.test_dataloaders[0].dataset.c1_mapper.items()
        used_c2 = trainer.test_dataloaders[0].dataset.c2_mapper.items()
        used_c3 = trainer.test_dataloaders[0].dataset.c3_mapper.items()
        inverted_mappers =   ({str(v): k  for k,v in used_c1 } ,
                                {str(v): k  for k,v in used_c2 },
                                {str(v): k  for k,v in used_c3 })
        
        c1,c2,c3 = batch['class1'], batch['class2'], batch['class3']
        c1_hat,c2_hat,c3_hat = pl_module(batch)
        
        c1_hat = F.softmax(c1_hat, dim=1)
        c2_hat = F.softmax(c2_hat, dim=1)
        c3_hat = F.softmax(c3_hat, dim=1)
        # print(classification, 'vs', classification_hat)
        c1_hat = torch.argmax(c1_hat,dim=1)
        c2_hat = torch.argmax(c2_hat,dim=1)
        c3_hat = torch.argmax(c3_hat,dim=1)
        # print(classification, 'vs', classification_hat)

        self.total_gt1.append(c1.cpu().numpy())
        self.total_predictions1.append(np.expand_dims(c1_hat.cpu().numpy(),axis=1))
        self.total_gt2.append(c2.cpu().numpy())
        self.total_predictions2.append(np.expand_dims(c2_hat.cpu().numpy(),axis=1))
        self.total_gt3.append(c3.cpu().numpy())
        self.total_predictions3.append(np.expand_dims(c3_hat.cpu().numpy(),axis=1))

        self.total_angular_excentricity.append(batch['angular_excentricity'].cpu().numpy())


    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        used_c1 = trainer.test_dataloaders[0].dataset.c1_mapper
        c1_names  = list(used_c1.keys())
        used_c2 = trainer.test_dataloaders[0].dataset.c2_mapper
        c2_names  = list(used_c2.keys())
        used_c3 = trainer.test_dataloaders[0].dataset.c3_mapper
        c3_names  = list(used_c3.keys())

        y1_true = np.concatenate(self.total_gt1, axis=0).ravel()
        y1_pred = np.concatenate(self.total_predictions1, axis=0).ravel()
        y2_true = np.concatenate(self.total_gt2, axis=0).ravel()
        y2_pred = np.concatenate(self.total_predictions2, axis=0).ravel()
        y3_true = np.concatenate(self.total_gt3, axis=0).ravel()
        y3_pred = np.concatenate(self.total_predictions3, axis=0).ravel()

        if self.output_dir is not None:
            # dump the predictions
            pd.DataFrame({'y1_true':y1_true,'y1_pred':y1_pred}).to_csv(self.output_dir / 'y1.csv')
            pd.DataFrame({'y2_true':y2_true,'y2_pred':y2_pred}).to_csv(self.output_dir / 'y2.csv')
            pd.DataFrame({'y3_true':y3_true,'y3_pred':y3_pred}).to_csv(self.output_dir / 'y3.csv')
            # dump the angular excentricity
            pd.DataFrame({'angular_excentricity':np.concatenate(self.total_angular_excentricity, axis=0).ravel()}).to_csv(self.output_dir / 'angular_excentricity.csv')
            # use json to dump the mappers
            tmp = {'used_c1': used_c1 ,'used_c2': used_c2, 'used_c3': used_c3}
            with open(self.output_dir / 'mappers.json', 'w') as fp:
                json.dump(tmp, fp)

        conf_mat1 = confusion_matrix(y1_true,y1_pred, normalize=self.normalize )
        conf_mat2 = confusion_matrix(y2_true,y2_pred, normalize=self.normalize )
        conf_mat3 = confusion_matrix(y3_true,y3_pred, normalize=self.normalize )

        fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(6*3,4),dpi=200)

        disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_mat1,
                                        display_labels=c1_names,
                                        )        
        disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_mat2,
                                        display_labels=c2_names,
                                        )        
        disp3 = ConfusionMatrixDisplay(confusion_matrix=conf_mat3,
                                        display_labels=c3_names,
                                        )
                                    
        disp1.plot(ax=ax[0])
        ax[0].xaxis.tick_top()
        ax[0].xaxis.set_label_position('top') 
        
        disp2.plot(ax=ax[1])
        ax[1].xaxis.tick_top()
        ax[1].xaxis.set_label_position('top') 
        
        disp3.plot(ax=ax[2])
        ax[2].xaxis.tick_top()
        ax[2].xaxis.set_label_position('top') 
        
        fig.show()
        if self.output_dir is not None:
            fig.savefig(os.path.join(self.output_dir,'confusion_matrix.png'))

        all_ang_excentricity = np.concatenate(self.total_angular_excentricity, axis=0).ravel()

        n_bins = 11
        mult = 90
        bins = np.linspace(0, 1*mult, n_bins, endpoint=True)

        f,a = plt.subplots(nrows=1,ncols=1,figsize=(6,4),dpi=200)
        a.hist(all_ang_excentricity*mult, bins=bins, ec='black', label='All')
        a.set_xlabel('Angular Distance [Deg]')
        a.set_ylabel('Count')
        f.show()

        if self.output_dir is not None:
            f.savefig(os.path.join(self.output_dir,f'AngularDistance_hist_{n_bins}bins.png'))

        # first plot is the histogram of correclty classified first Character along angular excentricity
        correct_c1 = np.where(y1_true == y1_pred)[0]
        correct_c1_ang = all_ang_excentricity[correct_c1]
        wrong_c1 = np.where(y1_true != y1_pred)[0]
        wrong_c1_ang = all_ang_excentricity[wrong_c1]
        
        # second plot is the histogram of correclty classified second Character along angular excentricity
        correct_c2 = np.where(y2_true == y2_pred)[0]
        correct_c2_ang = all_ang_excentricity[correct_c2]
        wrong_c2 = np.where(y2_true != y2_pred)[0]
        wrong_c2_ang = all_ang_excentricity[wrong_c2]

        # third plot is the histogram of correclty classified third Character along angular excentricity
        correct_c3 = np.where(y3_true == y3_pred)[0]
        correct_c3_ang = all_ang_excentricity[correct_c3]
        wrong_c3 = np.where(y3_true != y3_pred)[0]
        wrong_c3_ang = all_ang_excentricity[wrong_c3]

        bins_c_c1 = np.histogram(correct_c1_ang, bins=bins)[0]
        bins_w_c1 = np.histogram(wrong_c1_ang, bins=bins)[0]
        bins_c_c2 = np.histogram(correct_c2_ang, bins=bins)[0]
        bins_w_c2 = np.histogram(wrong_c2_ang, bins=bins)[0]
        bins_c_c3 = np.histogram(correct_c3_ang, bins=bins)[0]
        bins_w_c3 = np.histogram(wrong_c3_ang, bins=bins)[0]

        ratio_c1 = bins_c_c1 / (bins_c_c1 + bins_w_c1)
        ratio_c2 = bins_c_c2 / (bins_c_c2 + bins_w_c2)
        ratio_c3 = bins_c_c3 / (bins_c_c3 + bins_w_c3)

        f,a = plt.subplots(nrows=1,ncols=2,figsize=(6*2,4),dpi=200)
        b_x = np.array([(bins[i]+ bins[i+1])/2 for i in range(len(bins)-1)])
        n,_,_ =a[0].hist(all_ang_excentricity, bins=bins, ec='black', )
        a[0].set_xlim([-0.1,1])
        a[0].set_yticks(np.arange(0,np.max(n)+20,10))
        a[0].set_xticks(np.arange(0,1,.1))
        a[1].plot(b_x, ratio_c1, label='Z character')
        a[1].plot(b_x,ratio_c2, label='p character')
        a[1].plot(b_x,ratio_c3, label='c character')
        a[1].set_xlabel('Angular Distance [Deg]')
        a[1].set_ylabel('Correct Classification Ratio')
        a[1].legend()
        a[1].set_ylim([0,1])
        a[1].set_xlim([-0.1,1])
        a[0].grid(linestyle='--', linewidth=0.5, alpha=0.5)
        a[1].grid(linestyle='--', linewidth=0.5, alpha=0.5)
        a[1].set_yticks(np.arange(0,1,.1))
        a[1].set_xticks(np.arange(0,1,.1))
        # a[1].grid()
        f.show()
        if self.output_dir is not None:
            f.savefig(os.path.join(self.output_dir,f'CorrectRatio_{n_bins}bins.png'))

        fig2,ax2 = plt.subplots(nrows=1,ncols=3,figsize=(6*3,4),dpi=200)
        c_c1_n, c_c1_b, _ = ax2[0].hist([wrong_c1_ang,correct_c1_ang], bins=bins, alpha=0.5,color=['red','green'], stacked=True)
        w_c1_n, c_c1_n = c_c1_n[0], c_c1_n[1]
        c_c2_n, c_c2_b, _ = ax2[1].hist([wrong_c2_ang,correct_c2_ang], bins=bins, alpha=0.5,color=['red','green'], stacked=True)
        w_c2_n, c_c2_n = c_c2_n[0], c_c2_n[1]
        c_c3_n, c_c3_b, _ = ax2[2].hist([wrong_c3_ang,correct_c3_ang], bins=bins, alpha=0.5,color=['red','green'],  stacked=True)
        w_c3_n, c_c3_n = c_c3_n[0], c_c3_n[1]

        ax2[0].set_title('Z Character')
        ax2[1].set_title('p Character')
        ax2[2].set_title('c Character')
        ax2[0].set_xlabel('Angular Excentricity')
        ax2[1].set_xlabel('Angular Excentricity')
        ax2[2].set_xlabel('Angular Excentricity')
        ax2[0].set_ylabel('Count')
        ax2[1].set_ylabel('Count')
        ax2[2].set_ylabel('Count')
        ax2[0].legend()
        ax2[1].legend()
        ax2[2].legend()
        fig2.show()
        if self.output_dir is not None:
            fig2.savefig(os.path.join(self.output_dir,f'PerChar_hist_{n_bins}bins.png'))


        r_c1 = c_c1_n / (c_c1_n + w_c1_n)
        r_c2 = c_c2_n / (c_c2_n + w_c2_n)
        r_c3 = c_c3_n / (c_c3_n + w_c3_n)
        
        fig3,ax3 = plt.subplots(nrows=1,ncols=3,figsize=(6*3,4),dpi=200)
        ax3[0].plot(c_c1_b[1:],r_c1)
        ax3[1].plot(c_c2_b[1:],r_c2)
        ax3[2].plot(c_c3_b[1:],r_c3)
        ax3[0].set_title('Z Character')
        ax3[1].set_title('p Character')
        ax3[2].set_title('c Character')
        ax3[0].set_xlabel('Angular Excentricity')
        ax3[1].set_xlabel('Angular Excentricity')
        ax3[2].set_xlabel('Angular Excentricity')
        ax3[0].set_ylabel('Recall')
        ax3[1].set_ylabel('Recall')
        ax3[2].set_ylabel('Recall')
        # set minimum and maximum of y axis
        ax3[0].set_ylim([0,1])
        ax3[1].set_ylim([0,1])
        ax3[2].set_ylim([0,1])
        fig3.show()

        if self.output_dir is not None:
            fig3.savefig(os.path.join(self.output_dir,f'PerChar_CorrRatio_{n_bins}bins.png'))

        fig3,ax3 = plt.subplots(nrows=2,ncols=len(c1_names),figsize=(6*3,4),dpi=200)
        for i in range(len(c1_names)):
            ax3[0,i].set_title(f'Z:{c1_names[i]}')

            

            bins = np.linspace(0, 1, n_bins)

            # print(y1_true.shape)
            cur_char = np.where(y1_true == i)[0]
            
            correct_c1 = np.where((y1_true[cur_char] == y1_pred[cur_char]))[0]
            
            
            correct_c1_ang = all_ang_excentricity[correct_c1]
            wrong_c1 = np.where(y1_true[cur_char] != y1_pred[cur_char])[0]
            
            wrong_c1_ang = all_ang_excentricity[wrong_c1]

            c_c1_n, c_c1_b, _ = ax3[0,i].hist([wrong_c1_ang,correct_c1_ang], bins=bins, alpha=0.5, color=['red','green'], stacked=True)
            w_c1_n , c_c1_n = c_c1_n[0], c_c1_n[1]

            r_c1 = c_c1_n / (c_c1_n + w_c1_n)
            ax3[1,i].bar(bins[1:], r_c1, width=1.0/n_bins)
            ax3[1,i].set_ylim([0,1.1])
            ax3[1,i].set_xlim(ax3[0,i].get_xlim())
            
        fig3.show()

        if self.output_dir is not None:
            fig3.savefig(os.path.join(self.output_dir,f'Z_PerClass_histAndRatio_{n_bins}bins.png'))






class McIntoshClassificationFailureCasesCallback(pl.Callback):
    def __init__(self, focusMcIntoshChar, focusGTCharClass, focusPredCharClass, should_norm=False) -> None:

        super().__init__()

        self.should_norm = should_norm
        self.focusGTCharClass = focusGTCharClass
        self.focusPredCharClass = [focusPredCharClass] if isinstance(focusPredCharClass,str) else focusPredCharClass
        self.focusMcIntoshChar = focusMcIntoshChar

        assert self.focusMcIntoshChar in ["class1","class2","class3"]

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
        
        c1_mapper = trainer.test_dataloaders[0].dataset.c1_mapper
        c2_mapper = trainer.test_dataloaders[0].dataset.c2_mapper
        c3_mapper = trainer.test_dataloaders[0].dataset.c3_mapper
        mappers = {'class1':c1_mapper,'class2':c2_mapper,'class3':c3_mapper,}

        used_c1 = c1_mapper.items()
        used_c2 = c2_mapper.items()
        used_c3 = c3_mapper.items()
        inverted_mappers =   ({str(v): k  for k,v in used_c1 } ,
                                {str(v): k  for k,v in used_c2 },
                                {str(v): k  for k,v in used_c3 })
        
        c1,c2,c3 = batch['class1'], batch['class2'], batch['class3']
        McIntosh = c1,c2,c3
        gt = {'class1':c1.squeeze(1),'class2':c2.squeeze(1),'class3':c3.squeeze(1)}

        c1_hat,c2_hat,c3_hat = pl_module(batch)
        
        c1_hat_sm = F.softmax(c1_hat, dim=1)
        c2_hat_sm = F.softmax(c2_hat, dim=1)
        c3_hat_sm = F.softmax(c3_hat, dim=1)
        # print(classification, 'vs', classification_hat)
        c1_hat_class = torch.argmax(c1_hat_sm,dim=1)
        c2_hat_class = torch.argmax(c2_hat_sm,dim=1)
        c3_hat_class = torch.argmax(c3_hat_sm,dim=1)
        # print(classification, 'vs', classification_hat)
        McIntosh_hat = c1_hat_class,c2_hat_class,c3_hat_class
        pred =  {'class1':c1_hat_class,'class2':c2_hat_class,'class3':c3_hat_class}

        image = batch['image'].cpu()
        if image.ndim == 3:
            image = image.unsqueeze(1)

        if 'mask_one_hot' in batch:
            mask = batch['mask_one_hot'].cpu()

        idx_failures = torch.argwhere(
                            torch.logical_and(
                                (gt[self.focusMcIntoshChar] != pred[self.focusMcIntoshChar]),
                                torch.logical_and(
                                    (gt[self.focusMcIntoshChar] == mappers[self.focusMcIntoshChar][self.focusGTCharClass]),
                                    (sum(pred[self.focusMcIntoshChar] == mappers[self.focusMcIntoshChar][i] for i in self.focusPredCharClass).bool())
                                    )
                                )
                            )
        
        num_failures = idx_failures.nelement()
        
        # raise ValueError
        if num_failures>0:
            bs = num_failures

            if image.ndim == 4:

                
                if 'mask_one_hot' in batch:
                    fig, ax = plt.subplots(nrows=image.shape[1]+ mask.shape[1]-1 ,ncols=bs, 
                                        figsize=(3*bs, 3*image.shape[1]+mask.shape[1] ),
                                        facecolor='w', dpi=200)
                else:
                    
                    fig, ax = plt.subplots(nrows=2,ncols=bs, 
                                        figsize=(3*bs, 3*2),
                                        facecolor='w', dpi=200)
                
                
                for i in range(bs):
                    idx = int(idx_failures[i]) if num_failures>1 else int(idx_failures)
                    img = image[idx, 0, :, :].cpu().numpy()

                    # print(np.unique(img))

                    if 'mask_one_hot' in batch:
                        img_pen = mask[idx, 1, :, :].cpu().numpy()
                        img_um = mask[idx, 2, :, :].cpu().numpy()

                    g_C1 = str(int(gt['class1'][idx].cpu().numpy()))
                    g_C1 = inverted_mappers[0][g_C1]
                    g_C1 = 'SG' if g_C1 == 'SuperGroup' else g_C1
                    g_C2 = str(int(gt['class2'][idx].cpu().numpy()))
                    g_C2 = inverted_mappers[1][g_C2]
                    g_C3 = str(int(gt['class3'][idx].cpu().numpy()))
                    g_C3 = inverted_mappers[2][g_C3]


                    pred_C1 = str(pred['class1'][idx].cpu().numpy())
                    pred_C1 = inverted_mappers[0][pred_C1] 
                    pred_C1 = 'SG' if pred_C1 == 'SuperGroup' else pred_C1
                    pred_C2 = str(pred['class2'][idx].cpu().numpy())
                    pred_C2 = inverted_mappers[1][pred_C2] 
                    pred_C3 = str(pred['class3'][idx].cpu().numpy())
                    pred_C3 = inverted_mappers[2][pred_C3]

                    strMcIntosh = f'{g_C1}_{g_C2}_{g_C3}'
                    strMcIntosh_hat = f'{pred_C1}_{pred_C2}_{pred_C3}'

                
                    normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)


                    print(batch['group_name'][idx])

                    if num_failures>1:



                        ax[0,i].set_title(f"{batch['group_name'][idx]}")
                        ax[0,i].set_xlabel(f"GT: {strMcIntosh}\nPred:{strMcIntosh_hat}")
                        
                        
                        if self.should_norm:
                            ax[0,i].imshow(3500*img, cmap='gray', norm = normalize, interpolation='none')
                        else:
                            ax[0,i].imshow(img, cmap='gray', norm = normalize, interpolation='none')
                        # ax[0,i].imshow(img, cmap='gray')
        
                        ax[0,i].set_xticks([])
                        ax[0,i].set_yticks([])
                        if 'mask_one_hot' in batch:
                            ax[1,i].imshow(img_pen, cmap='gray', interpolation='none')
                            ax[2,i].imshow(img_um, cmap='gray', interpolation='none')
                            ax[1,i].set_xticks([])
                            ax[2,i].set_xticks([])
                            ax[1,i].set_yticks([])
                            ax[2,i].set_yticks([])
                            # ax[0,i].axis('off')
                            # ax[1,i].axis('off')
                            # ax[2,i].axis('off')
                    else:
                        ax[0].set_title(f"{batch['group_name'][idx]}")
                        ax[0].set_xlabel(f"GT: {strMcIntosh}\nPred:{strMcIntosh_hat}")


                        if self.should_norm:
                            ax[0].imshow(3500*img, cmap='gray', norm = normalize, interpolation='none')
                        else:
                            ax[0].imshow(img, cmap='gray', norm = normalize, interpolation='none')
                        
                        ax[0].set_xticks([])
                        ax[0].set_yticks([])

                        if 'mask_one_hot' in batch:
                            ax[1].imshow(img_pen, cmap='gray', interpolation='none')
                            ax[2].imshow(img_um, cmap='gray', interpolation='none')
                            ax[1].set_xticks([])
                            ax[2].set_xticks([])
                            ax[1].set_yticks([])
                            ax[2].set_yticks([])
                       
                plt.subplots_adjust(hspace=0.35)
                plt.show()




def display_first_channel_batchMcIntosh(batch, image_key, predictions, gt, mappers):
    image = batch[image_key].cpu()
    bs = image.shape[0]
    
    show_classes = ['0','1']
    print(mappers)
    log.debug(image.ndim)
    # print(mapper)
    # print('1')
    if image.ndim == 4:
        # print('2')
        for i in range(bs):
            # img = 3500* image[i, 0, :, :].cpu().numpy()
            img =  image[i, 0, :, :].cpu().numpy()

            pred_C1 = str(predictions[0][i].cpu().numpy())
            g_C1 = str(gt[0][i].cpu().numpy()[0])
            pred_C2 = str(predictions[1][i].cpu().numpy())
            g_C2 = str(gt[1][i].cpu().numpy()[0])
            pred_C3 = str(predictions[2][i].cpu().numpy())
            g_C3 = str(gt[2][i].cpu().numpy()[0])
            
            if (pred_C1 in show_classes) and (g_C1 in show_classes) and (g_C1 != pred_C1):
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(3, 3),  dpi=200)

                normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)
                ax.imshow(img, cmap='gray', norm = normalize)
                
                strMcIntosh = mappers[0][g_C1] + mappers[1][g_C2] + mappers[2][g_C3]
                strMcIntosh_hat = mappers[0][pred_C1] + mappers[1][pred_C2] + mappers[2][pred_C3]
                ax.set_title(f"GT: {strMcIntosh}, Pred: {strMcIntosh_hat}")
                plt.show()
                plt.close(fig)

        