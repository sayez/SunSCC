from contextlib import AbstractContextManager
import zipfile
from sunscc.dataset.transform.pipelines import Compose
import collections
from functools import partial
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
import logging
import numpy as np
import cv2
import os
from hydra.utils import call, instantiate
import skimage.io as io

import random

from .utils import *
from astropy.io import fits
import matplotlib.pyplot as plt

from .utils import *
from astropy.io import fits
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)

class DeepsunSegmentationDataset(Dataset):
    """ A dataset containing a directory per type (image, segmentation1, segmetnation2...)

        Every type directory contains an image/target file per sample , the 
        number of files per directory must be the same, as must be the names of
        the files.

        Args:
            root_dir: directory containing the set directories.
            partition: subdirectory inside the root_dir (train, test or val).
            dtypes: the types that must be existing directories.
            transforms: a callable, a dict with _target_ or a list of dicts with
                _target_'s the list will be passed through a custom Compose method.
    
    """
    def __init__(
        self, root_dir, partition, dtypes, transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.abc.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.abc.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        self.transforms = transforms
        print(os.getcwd())
        self.root_dir = Path(root_dir) / partition
        self.dtypes = dtypes

        self.main_dtype = dtypes[0]
        self.target_types = dtypes[1:]

        self.files = []
        self.masks_lists = { t: sorted((self.root_dir / t).iterdir()) for t in self.target_types}

        # for i, file in enumerate(sorted((self.root_dir / self.main_dtype).iterdir())[:]):
        for i, file in enumerate(sorted((self.root_dir / self.main_dtype).iterdir())):
            cur = {}
            cur[self.main_dtype] = file
            cur['name'] = os.path.basename(file)
            tmp1 = os.path.splitext(cur['name'])[0]
            for t in self.target_types:
                # print(len(sorted((self.root_dir / self.main_dtype).iterdir())))
                # print(t)
                cur[t] = self.masks_lists[t][i]
                tmp2 = os.path.splitext(os.path.basename(cur[t]))[0]
                # print(i)
                # print(tmp1, "    ", tmp2)
                assert tmp1 == tmp2

            self.files.append(cur)



    def __len__(self) -> int:
        return len(self.files)


    def __getitem__(self, index: int, do_transform=True):
        sample = {} # dictionary with 'image', "segmentation" entries

        img_name =  self.files[index]["image"]

        basename =os.path.basename(img_name).split('.')[0]
        sample['name'] = basename

        hdulst:fits.HDUList = fits.open(img_name)
        image = hdulst[0]
        header = image.header
        center = np.array(image.shape)//2
        radius = header['SOLAR_R']
        sample['solar_disk'] = create_circular_mask( image.shape[0], image.shape[1] ,center,radius)
        
        sample["image"] = (image.data).astype(float) # load image from directory with skimage
        sample["segmentation"] = np.array([io.imread(self.files[index][t]).astype(float) 
                                    for t in self.target_types ])
        
        sample['sample_id'] = f'{index}'

        
        if self.transforms is not None and do_transform:
            
            sample = self.transforms(**sample)
            # print(sample.keys())
            # print(np.array(sample["segmentation"]).shape)

        hdulst.close()
        return sample


class DeepsunSegmentationTestDataset(Dataset):
    def __init__(
        self, root_dir, partition, dtypes, patch_side, transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.abc.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.abc.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        self.transforms = transforms
        self.root_dir = Path(root_dir) / partition
        self.dtypes = dtypes

        self.main_dtype = dtypes[0]
        self.target_types = dtypes[1:]

        self.files = []
        self.masks_lists = { t: sorted((self.root_dir / t).iterdir()) for t in self.target_types}

        for i, file in enumerate(sorted((self.root_dir / self.main_dtype).iterdir())):
            cur = {}
            cur[self.main_dtype] = file
            cur['name'] = os.path.basename(file)
            tmp1 = os.path.splitext(cur['name'])[0]
            for t in self.target_types:
                
                cur[t] = self.masks_lists[t][i]
                tmp2 = os.path.splitext(os.path.basename(cur[t]))[0]
                
                assert tmp1 == tmp2

            self.files.append(cur)

        self.patch_side = patch_side
        self.grid_size = self.get_grid_size(self.files[0]['image'], self.patch_side)



    def get_grid_size(self, img_fn, patch_side):
        side = io.imread(str(img_fn)).shape[0] // patch_side
        return side*side

    def __len__(self) -> int:
        
        return len(self.files)* self.grid_size

    def crop_patch(self, img, patch_index):
        i = patch_index // int(np.sqrt(self.grid_size))
        j = patch_index % int(np.sqrt(self.grid_size))
        return img.copy()[
                            i*self.patch_side: (i+1)*self.patch_side,
                            j*self.patch_side: (j+1)*self.patch_side
                        ]


    def __getitem__(self, index: int, do_transform=True):
        # print(do_transform)
        sample = {} 
        idx_img = index // self.grid_size
        idx_patch = index % self.grid_size

        img_name =  self.files[idx_img]["image"]
        hdulst:fits.HDUList = fits.open(img_name)
        image = hdulst[0]
        header = image.header
        center = np.array(image.shape)//2
        radius = header['SOLAR_R']
        sample['solar_disk'] = create_circular_mask( image.shape[0], image.shape[1] ,center,radius)
        
        sample["image"] = self.crop_patch((io.imread(img_name)).astype(float), idx_patch) # load image from directory with skimage
        

        sample["segmentation"] = np.array([self.crop_patch(io.imread(self.files[idx_img][t]), idx_patch ).astype(float)
                                    for t in self.target_types ])


        sample['sample_id'] = f'{idx_img}_{idx_patch}'
        # print("Avant", sample["segmentation"].shape)
        if self.transforms is not None and do_transform:
            
            sample = self.transforms(**sample)

        #####################################
        sample["segmentation"] = [sample["segmentation"][0]] if len(sample["segmentation"])>0 else np.zeros_like(sample["image"])
        
        hdulst.close()

        return sample

class DeepsunSegmentationTTA_TestDataset(Dataset):
    def __init__(
        self, root_dir, partition, dtypes, patch_side, num_tta=0, transforms=None, tta_transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.abc.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.abc.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        self.transforms = transforms
        self.root_dir = Path(root_dir) / partition
        self.dtypes = dtypes

        self.main_dtype = dtypes[0]
        self.target_types = dtypes[1:]

        self.files = []
        self.masks_lists = { t: sorted((self.root_dir / t).iterdir()) for t in self.target_types}


        for i, file in enumerate(sorted((self.root_dir / self.main_dtype).iterdir())):
            cur = {}
            cur[self.main_dtype] = file
            cur['name'] = os.path.basename(file)
            tmp1 = os.path.splitext(cur['name'])[0]
            for t in self.target_types:
                cur[t] = self.masks_lists[t][i]
                tmp2 = os.path.splitext(os.path.basename(cur[t]))[0]
                
                assert tmp1 == tmp2

            self.files.append(cur)

        self.patch_side = patch_side
        self.grid_size = self.get_grid_size(self.files[0]['image'], self.patch_side)

        self.num_tta = num_tta if num_tta > 0 else 1



    def get_grid_size(self, img_fn, patch_side):
        side = io.imread(str(img_fn)).shape[0] // patch_side
        return side*side

    def __len__(self) -> int:
        return len(self.files)* self.grid_size * self.num_tta
    

    def crop_patch(self, img, patch_index):
        i = patch_index // int(np.sqrt(self.grid_size))
        j = patch_index % int(np.sqrt(self.grid_size))
        return img.copy()[
                            i*self.patch_side: (i+1)*self.patch_side,
                            j*self.patch_side: (j+1)*self.patch_side
                        ]


    def __getitem__(self, index: int, do_transform=True):
        # print(do_transform)
        sample = {} # dictionary with 'image', "segmentation" entries

        idx_img = index // (self.grid_size * self.num_tta) 
        # idx_img += 2000
        idx_patch_tta = index % (self.grid_size * self.num_tta)

        idx_patch = idx_patch_tta // self.num_tta
        idx_aug = idx_patch_tta % self.num_tta

        sample['TTA_idx'] = idx_aug

        img_name =  self.files[idx_img]["image"]
        # try:

        hdulst:fits.HDUList = fits.open(img_name)
        image = hdulst[0]
        header = image.header
        center = np.array(image.shape)//2
        radius = header['SOLAR_R']
        # sample['solar_disk'] = create_circular_mask( image.shape[0], image.shape[1] ,center,radius)
        basic_mask = create_circular_mask( image.shape[0], image.shape[1] ,center,radius=radius*1.03)
        sample['solar_disk'] = get_sun_mask( image.data, basic_mask, radius)
        
        sample["image"] = (image.data).astype(float)# load image from directory with skimage
        
        sample["segmentation"] = np.array([io.imread(self.files[idx_img][t]).astype(float)
                                    for t in self.target_types ]) if len(self.target_types) > 0 else np.array([np.zeros_like(image.data).astype(float)])

        sample['sample_id'] = f'{idx_img}_{idx_patch}_{idx_aug}'
        basename = os.path.basename(img_name).split(".")[0]
        sample['sample_id2'] = f'{basename}_{idx_patch}_{idx_aug}'

        sample['aug_history'] = []
        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)
        
        hdulst.close()
        return sample