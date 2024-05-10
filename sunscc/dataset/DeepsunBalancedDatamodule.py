import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from numpy.lib.arraysetops import isin
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sunscc.dataset import PrepareStrategy, SetupStrategy
from hydra.utils import instantiate

from torch.utils.data.sampler import WeightedRandomSampler

from tqdm.notebook import tqdm
import time

from .datamodule import SunSCCDataModule

log = logging.getLogger(__name__)

# class SunSCCBalancedDataModule(pl.LightningDataModule):
class SunSCCBalancedDataModule(SunSCCDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        train_dataset: Optional[dict] = None,
        val_dataset: Optional[dict] = None,
        test_dataset: Optional[dict] = None,
        strategies: Optional[Dict[str, Union[PrepareStrategy, SetupStrategy]]] = None,
        batch_size: int = 2,
        num_workers: int = 1,
        balanced = True,
        char_to_balance = 'class1', # 'class1' or 'class2' or 'class3'
        **kwargs,
    ):
        # super().__init__()
        super().__init__(data_dir,
                            dataset_name,
                            train_dataset,
                            val_dataset,
                            test_dataset,
                            strategies,
                            batch_size,
                            num_workers)
        
        self.data_dir = Path(data_dir) / dataset_name
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.balanced = balanced
        self.char_to_balance = char_to_balance

    def train_dataloader(self) -> DataLoader:

        print('Balancing weights of character {}...'.format(self.char_to_balance))
        st = time.time()
        class_weights = [(1/self.train_ds.class_distrib[self.char_to_balance][i]) 
                            for i in range(len(self.train_ds.class_distrib[self.char_to_balance].keys()))]
        print('Class weights: {}'.format(class_weights))
        print('Time to calculate weights: {}'.format(time.time() - st))

        print('Calculating weights for each sample...')
        st = time.time()
        sample_weights = [0] * len(self.train_ds)
        for idx,sample in tqdm(enumerate(self.train_ds), leave=False, total=len(self.train_ds)):
            sample_weights[idx] = class_weights[sample[self.char_to_balance][0]]
        print('Time to calculate weights for each sample: {}'.format(time.time() - st))
            
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # shuffle=True,
            sampler=WeightedRandomSampler(sample_weights, len(self.train_ds), replacement=True)
        )

    def val_dataloader(self) -> DataLoader:
        # class_weights = [(1/self.train_ds.class_distrib[i]) 
        #                     for i in range(len(self.train_ds.class_distrib.keys()))]
        # sample_weights = [0] * len(self.val_ds)
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
            # # strongly advised not to shuffle/sample for validation / test / predict
            # sampler=WeightedRandomSampler(sample_weights, len(self.val_ds), replacement=True)
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
        )
