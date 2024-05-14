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


from ..dataset.utils import create_circular_mask
from astropy.io import fits

log = logging.getLogger(__name__)


class PlotImageCallback(pl.Callback):
    def __init__(self, num_samples=32):
        self.num_samples = num_samples
        super().__init__()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        dm = trainer.val_dataloaders[0]
        seen = 0
        for batch in dm:
            segmentation = pl_module.predict(batch)
            if seen > self.num_samples:
                break
            cat_segm = to_categorical(segmentation)
            bs = cat_segm.shape[0]
            batch["_segm"] = cat_segm
            display_batch(batch, "image", "_segm")
            seen += bs


class InputHistoCallback(pl.Callback):
    def __init__(self, show_percentage=0.1, inputs=("image",), val=False) -> None:
        self.show = show_percentage
        self.inputs = inputs
        self.rng = np.random.default_rng()
        self.shown = False
        self.val = val

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.on_train_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.rng.random() > self.show:
            return
        fig, axs = plt.subplots(len(self.inputs), figsize=(10, 3), squeeze=False)
        for ax, input in zip(axs[0], self.inputs):
            flat_batch = batch[input].cpu().flatten().to(torch.uint8).numpy()
            ax.hist(flat_batch, bins=256, range=(0, 255))
        plt.show()
        plt.close(fig)


class PlotTrainCallback(pl.Callback):
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
            if torch.is_tensor(batch[dtype]):
                batch[dtype] = batch[dtype].squeeze(1).to(torch.float)
        segmentation = pl_module.predict(batch) 
        # segmentation = pl_module.predict(torch.squeeze(batch,1))       
        # print("on_train_batch_end: ", segmentation.shape, torch.unique(segmentation[0]))
        # segmentation = F.softmax(segmentation, dim=1)
        # print("on_train_batch_end AFTER SOFTMAX: ", segmentation.shape, torch.unique(segmentation[0]))

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




class SavePredictionMaskCallback(pl.Callback):
    def __init__(self, output_dir, max_batch_size) :
        self.output_dir = Path(output_dir)
        self.max_batch_size = max_batch_size
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        # if trainer.running_sanity_check:
        #     return    
        if trainer.sanity_checking:
            return
        
        batch["segmentation"] = batch["segmentation"][0]
        
        segmentation = pl_module(batch)

        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()

        dataset = trainer.datamodule.test_ds
        
        cur_batch_size = batch["segmentation"].shape[0]

        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i
            name = (dataset.files[idx]["name"]).split(".")[0]+ ".png"

            io.imsave(self.output_dir / name, segmentation[i], check_contrast=False)

def reconstruct_segmentation_image(grid_size, batch, outputs, output_dir: Path):
        segmentation = outputs
        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()   
        cur_batch_size = batch["segmentation"].shape[0]

        assert cur_batch_size == grid_size
        
        name = (batch["name"][0]).split(".")[0]+ ".png"

        side = int(segmentation[0].shape[0] * np.sqrt(grid_size))
        # print(side)
        dump = np.zeros((side,side)).astype(np.uint8)
        for j, patch in enumerate(segmentation):
            a = j // int(np.sqrt(grid_size))
            b = j % int(np.sqrt(grid_size))
            dump[
                    a*patch.shape[0]: (a+1)*patch.shape[0],
                    b*patch.shape[1]: (b+1)*patch.shape[1]
                ] = patch
        
        io.imsave(output_dir / name, dump, check_contrast=False)

class SavePredictionMaskCallback2(pl.Callback):
    def __init__(self, output_dir, max_batch_size) :
        self.output_dir = Path(output_dir)
        print(self.output_dir.stem)
        self.max_batch_size = max_batch_size
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reconstructed_image = []

    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
    # def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
        
        # print(f"SavePredictionMaskCallback2.on_predict_batch_end:")

        # batch["segmentation"] = batch["segmentation"]
        # for dtype in batch:
        #     # print(batch[dtype].shape)
        #     batch[dtype]=torch.squeeze(batch[dtype],1)
        # segmentation = pl_module(batch)

        segmentation = outputs
        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()

        dataset = trainer.datamodule.test_ds
        grid_size = dataset.grid_size
        
        cur_batch_size = batch["segmentation"].shape[0]


        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i

            self.reconstructed_image.append(segmentation[i])
            # print(len(self.reconstructed_image))
            if len(self.reconstructed_image) == grid_size:
                index = idx // grid_size
                # name = (batch["image_name"][0]).split(".")[0]+ ".png"
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
                
                io.imsave(self.output_dir / name, dump, check_contrast=False)
                self.reconstructed_image = []
            
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        if trainer.sanity_checking:
            return
        
        batch["segmentation"] = batch["segmentation"][0]
        
        batch.pop('sample_id',None)

        segmentation = pl_module(batch)

        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()

        dataset = trainer.datamodule.test_ds
        grid_size = dataset.grid_size
        
        cur_batch_size = batch["segmentation"].shape[0]


        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i

            self.reconstructed_image.append(segmentation[i])
            # print(len(self.reconstructed_image))
            if len(self.reconstructed_image) == grid_size:
                index = idx // grid_size
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
                solar_disk = create_circular_mask( image.shape[0], image.shape[1] ,center,radius)
                
                dump = dump * solar_disk # keep only predictions inside solar disk
                dump[image == 0] = 0  # Removes predictions where solar disk is cut (telescope image was shifted so sun is centered -> zero padding)

                
                # print(np.unique(dump))
                # print(self.output_dir)
                io.imsave(self.output_dir / name, dump, check_contrast=False)
                self.reconstructed_image = []
            
            # name = (dataset.files[idx]["name"]).split(".")[0]+ ".png"
            # io.imsave(self.output_dir / name, segmentation[i], check_contrast=False)
            
def write_image(filename, image):
    filename.parent.mkdir(parents=True, exist_ok=True)
    res = cv2.imwrite(str(filename), image)
    if res is False:
        raise RuntimeError("CV2 imwrite failed to save image.")


