from email.mime import base
from tokenize import group
from bioblue.dataset.transform.pipelines import Compose
import collections
from functools import partial

from torch.utils.data import Dataset, DataLoader
import os
from hydra.utils import call, instantiate
from pathlib import Path
import skimage.io as io
import json

import numpy as np

from tqdm.notebook import tqdm

import traceback
import time

from albumentations.core.transforms_interface import DualTransform

from astropy.io import fits
from bioblue.dataset.utils import *

from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from copy import deepcopy

import pickle

import concurrent.futures
from itertools import repeat
import multiprocessing


root_dir = "../datasets/Classification_dataset/2002-2019"

all_samples ={'train':{},'val':{},'test':{}}
for p in all_samples.keys():
    print('loading', p)
    st = time.time()    
    filename = os.path.join(root_dir,'test',f'all_samples_{p}.npy' )
    tmp = np.load(filename, allow_pickle=True).item()
    print('Elapsed time', time.time()-st)
    
    all_samples[p] = tmp

print("Dumping")
st = time.time()    
tot_npy_file = os.path.join(root_dir, 'test', f'all_samples.npy')
np.save(tot_npy_file, all_samples)
print('Elapsed time', time.time()-st)
