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
import ipywidgets as widgets
from IPython.display import display


from ..dataset.utils import create_circular_mask, get_sun_mask
from astropy.io import fits

from .DeepsunTTAReverseUtils import *

log = logging.getLogger(__name__)


class DeepsunPlotTrainCallback(pl.Callback):
    def __init__(self, show_percentage=0.1, input="image", val=False):
        self.show = show_percentage
        self.input = input
        self.rng = np.random.default_rng()
        self.shown = False
        self.val = val

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) :
        if self.val:
            self.on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
   
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.val:
            self.on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not self.shown:
            return
        log.debug(outputs)
        img = batch[self.input].cpu()
        bs = img.shape[0]
        
        for dtype in batch:
            # print(dtype)
            if torch.is_tensor(batch[dtype]) and batch[dtype].ndim > 2:
                batch[dtype] = batch[dtype].squeeze(1).to(torch.float)
        segmentation = pl_module.predict(batch) 

        segmentation = to_categorical(segmentation,argmax_dim=1).cpu()
        # print("on_train_batch_end AFTER to_categorical: ", segmentation.shape, torch.unique(segmentation[0]))

        # segmentation = to_categorical(segmentation).cpu()
        batch["_segm"] = segmentation
        display_batch(batch, self.input, "_segm")
        del batch["_segm"]  # really needed ?

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.val:
            self.on_train_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.val:
            self.on_train_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.rng.random() > self.show:
            self.shown = False
            return
        self.shown = True
        input_format = pl_module.input_format  # FIXME : create ABC for this
        for input_key in input_format:
            display_batch(batch, image_key=input_key)

        output_format = pl_module.output_format  # FIXME : as above
        for output_key in output_format:
            display_batch(batch, image_key=self.input, segm_key=output_key)





class DeepsunSavePredictionMaskTTACallback(pl.Callback):
    def __init__(self, output_dir, max_batch_size) :
        self.output_dir = Path(output_dir)
        # print(self.output_dir.stem)
        self.max_batch_size = max_batch_size
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)


        self.patch_augmentations = []
        self.patch_gt = []
        self.reconstructed_image = []
        self.reconstructed_image_proba_map = []


    def reverseTTA_mask(self,batch, seg_hat):
        
        transforms = batch['aug_history'].copy()
        # print("transforms", transforms)
        transforms.reverse()
        # print("transforms", transforms)

        rev_msk = seg_hat.clone()

        # print(augmented_images.shape)
        for transform in transforms:
            transform_name, params = list(transform.items())[0]
            # print(params)
            if transform_name == "DeepsunRandomShiftScaleRotate":
                # print("DeepsunRandomShiftScaleRotate", rev_msk.shape)
                rev_msk = reverse_batch_ShiftScaleRotate(rev_msk,params)

            if transform_name == 'DeepsunTTARotate90':
                # print("DeepsunTTARotate90")
                rev_msk = reverse_batch_Rotate90(rev_msk,params)
        
        return rev_msk

    def merge_augmentations(self, augmentations, void_label=-1):
        augmentations = torch.stack(augmentations,dim=0)
        rounded_msk = torch.round(augmentations)
        rounded_msk[rounded_msk==void_label] = torch.nan
        mode_msk = torch.mode(rounded_msk,dim=0,keepdims=False)[0]
        mode_msk = torch.nan_to_num(mode_msk)
        # print('mode uniques' ,torch.unique(mode_msk))
        return mode_msk

    def augmentations_proba_map(self, augmentations, void_label=-1):
        # print('augmentations[0] shape',augmentations[0].shape)
        augmentations = torch.stack(augmentations,dim=0)
        # print('augmentations uniques',torch.unique(augmentations))
        rounded_msk = torch.round(augmentations)
        # print('augmentations uniques',torch.unique(augmentations))
        rounded_msk[rounded_msk==void_label] = torch.nan

        average = torch.nanmean(rounded_msk,dim=0,keepdims=False)
        average = torch.nan_to_num(average)
        # print('average.shape',average.shape)

        return average



    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
        

        batch["segmentation"] = batch["segmentation"]
        for dtype in batch:
            # print(batch[dtype].shape)
            batch[dtype]=torch.squeeze(batch[dtype],1)

        segmentation = pl_module(batch)

        rev_segmentation = self.reverseTTA_mask(batch, segmentation)

        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()

        dataset = trainer.datamodule.test_ds
        grid_size = dataset.grid_size
        num_tta = dataset.num_tta
        
        cur_batch_size = batch["segmentation"].shape[0]
        


        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i
            print(idx)
            print(self.patch_augmentations)
            print(self.reconstructed_image)
            self.patch_augmentations.append(segmentation[i])

            if len(self.patch_augmentations) == num_tta:
                print('Merge_patch')

                
                # We have reversed all augmentations for the current patch to reconstruct
                # We can do the majority vote and append result to self.reconstructed_image
                # patch_result = merge_augmentations(self.patch_augmentations)
                patch_result = []
                self.reconstructed_image.append(patch_result)
                self.patch_augmentations = []
                
                pass

            # print(len(self.reconstructed_image))
            if len(self.reconstructed_image) == grid_size:
                print("Dump Image")
                index = idx // (grid_size*num_tta)
                name = (dataset.files[index]["name"]).split(".")[0]+ ".png"

                side = int(self.reconstructed_image[0].shape[0] * np.sqrt(grid_size))
                # print(side)
                dump = np.zeros((side,side)).astype(np.uint8)
                # for j, patch in enumerate(self.reconstructed_image):
                #     a = j // int(np.sqrt(grid_size))
                #     b = j % int(np.sqrt(grid_size))
                #     dump[
                #             a*patch.shape[0]: (a+1)*patch.shape[0],
                #             b*patch.shape[1]: (b+1)*patch.shape[1]
                #         ] = patch
                
                # print(np.unique(dump))
                # print(self.output_dir)
                io.imsave(self.output_dir / name, dump, check_contrast=False)
                self.reconstructed_image = []
            
            # name = (dataset.files[idx]["name"]).split(".")[0]+ ".png"
            # io.imsave(self.output_dir / name, segmentation[i], check_contrast=False)
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        if trainer.sanity_checking:
            return
        
        void_label = pl_module.loss.ignore_index
        neg_void_label = -1
        # print('void_label is ', void_label)

        segmentation = batch["segmentation"]

        # print('void_label: ',void_label, 'neg_void_label: ', neg_void_label)

        # print('segmentation shape',segmentation.shape)
        segmentation_hat = pl_module(batch)
        segmentation_hat = to_categorical(segmentation_hat)#.to(torch.uint8)


        # print('segmentation_hat shape',segmentation_hat.shape)
        # print("1",torch.unique(segmentation))
        segmentation_hat[segmentation == void_label] = neg_void_label
        # print("2",torch.unique(segmentation))
        segmentation_hat = self.reverseTTA_mask(batch,segmentation_hat)

        rev_gt = self.reverseTTA_mask(batch, batch["segmentation"].squeeze().clone())
        
        dataset = trainer.datamodule.test_ds
        grid_size = dataset.grid_size
        num_tta = dataset.num_tta
        
        cur_batch_size = batch["segmentation"].shape[0]
        # print('cur_batch_size', cur_batch_size)

        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i
            
            self.patch_augmentations.append(segmentation_hat[i])
            self.patch_gt.append(rev_gt[i].cpu().squeeze())

            if len(self.patch_augmentations) == num_tta:
                # print('Merge_patch')
                # We have reversed all augmentations for the current patch to reconstruct
                # We can do the majority vote and append result to self.reconstructed_image
                patch_result = self.merge_augmentations(self.patch_augmentations, void_label=void_label)
                
                patch_result[patch_result == neg_void_label] = 0
                self.reconstructed_image.append(patch_result)
                
                self.patch_augmentations_fg =[]
                # for p_aug in self.patch_gt:
                for p_aug in self.patch_augmentations:
                    # print(torch.unique(p_aug))
                    p_aug = p_aug.clone()
                    p_aug[ p_aug >1 ] = 1
                    # print(torch.unique(p_aug))

                    self.patch_augmentations_fg.append(p_aug)

                patch_proba_map = self.augmentations_proba_map(self.patch_augmentations_fg, void_label=void_label)
                patch_proba_map[patch_result == neg_void_label] = 0
                self.reconstructed_image_proba_map.append(patch_proba_map.clone())

                self.patch_augmentations = []
                self.patch_augmentations_fg = []
                self.patch_gt = []
                pass

            # print(len(self.reconstructed_image))
            if len(self.reconstructed_image) == grid_size:
                # print("Dump Image")
                index = idx // (grid_size*num_tta)
                name = (dataset.files[index]["name"]).split(".")[0]+ ".png"

                side = int(self.reconstructed_image[0].shape[0] * np.sqrt(grid_size))
                # print(side)
                dump = np.zeros((side,side)).astype(np.uint8)
                for j, patch in enumerate(self.reconstructed_image):
                    a = j // int(np.sqrt(grid_size))
                    b = j % int(np.sqrt(grid_size))
                    dump[
                            a*patch.shape[0]: (a+1)*patch.shape[0],
                            b*patch.shape[1]: (b+1)*patch.shape[1]
                        ] = patch

                hdulst:fits.HDUList = fits.open(dataset.files[index]["image"])
                image = hdulst[0]
                header = image.header
                center = np.array(image.shape)//2
                radius = header['SOLAR_R']

                # print('here - creating sun mask')
                basic_mask = create_circular_mask( image.shape[0], image.shape[1],center=center,radius=radius*1.03)
                solar_disk = get_sun_mask( image.data, basic_mask, radius)

                colors = plt.cm.jet(np.linspace(0, 1, 256))
                colors[0, :] = np.array([0, 0, 0, 0])
                colors = plt.matplotlib.colors.ListedColormap(colors)
                
                dump = dump * solar_disk # keep only predictions inside solar disk
                dump[image == 0] = 0  # Removes predictions where solar disk is cut (telescope image was shifted so sun is centered -> zero padding)

                
                io.imsave(self.output_dir / name, dump, check_contrast=False)
                self.reconstructed_image = []

                dump_proba_map = np.zeros((side,side)).astype(np.float32)
                for j, patch in enumerate(self.reconstructed_image_proba_map):
                    a = j // int(np.sqrt(grid_size))
                    b = j % int(np.sqrt(grid_size))
                    dump_proba_map[
                            a*patch.shape[0]: (a+1)*patch.shape[0],
                            b*patch.shape[1]: (b+1)*patch.shape[1]
                        ] = patch
                
                dump_proba_map = dump_proba_map * solar_disk # keep only predictions inside solar disk
                dump_proba_map[image == 0] = 0  # Removes predictions where solar disk is cut (telescope image was shifted so sun is centered -> zero padding)
                
                name_proba_map = (dataset.files[index]["name"]).split(".")[0]+ "_proba_map.npy"
                np.save(self.output_dir /name_proba_map,dump_proba_map)
                # io.imsave(self.output_dir / name_proba_map, dump_proba_map, check_contrast=False)
                self.reconstructed_image_proba_map = []
                