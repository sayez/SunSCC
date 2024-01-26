from sunscc.transforms import *
from pathlib import Path

import pytorch_lightning as pl
import torch

from omegaconf import open_dict
from hydra.utils import instantiate

import os
import sys
# Get the absolute path of the repository's root directory
module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the root directory to the Python path
sys.path.append(module_dir)

from typing import Mapping

#import deepcopy
from copy import deepcopy

from sunscc.nb.load import load_from_dir, load_from_dir2, load_from_cfg

import argparse
from datetime import datetime


def main(args):
    # # Old dataset with NO overlap: db_bbox touche 1 seul ms_bbox ET ce ms_bbox touche 1 seul db_bbox
    # outpath = 'results_old_dataset'
    # use_npy = 'test/all_samples.npy'
    # New dataset with overlap: db_bbox associe a la ms_bbox la plus proche, sans contrainte de distance/unicité
    # outpath = 'results_new_dataset'
    # use_npy = 'rebuttal/all_samples.npy'
    # New dataset with overlap: db_bbox associe a la ms_bbox la plus proche, sans contrainte de distance/unicité
    # outpath = 'results_overlap_only'
    # use_npy = 'rebuttal_overlap_only/all_samples.npy'
    # New dataset with overlap and close near limb:
    # outpath = 'results_filtered'
    # use_npy = 'rebuttal_filtered/all_samples.npy'
    outpath = args.outpath
    use_npy = args.use_npy
    

    print('LOADING RUN')
    st = datetime.now()
    run_dir = Path(args.run_dir)

    config, model, dm, trainer = load_from_dir2(
                                    run_path= run_dir,
                                    load_trainer=False,
                                    override = [
                                    f'++use_npy={use_npy}',  
                                    ]
                                )
    conf3 = deepcopy(config)

    conf3.model.parts_to_train = ['MLP3']

    conf3.trainer.max_epochs=100
    conf3.dataset.char_to_balance = 'class3'


    module, datamodule = load_from_cfg(conf3, recursive=False)
    
    conf3.logger = []
    
    et = datetime.now()
    print(f'LOADED RUN IN {et-st}')
    print('INITIALIZING RUN')
    st = datetime.now()

    callbacks = []
    if isinstance(conf3.callbacks, Mapping):
        conf3.callbacks = [cb for cb in conf3.callbacks.values()]
    for callback in conf3.callbacks:
        if callback['_target_'] == 'sunscc.callback.WandBCallback':
            print(type(callback))
    #         callback['cfg'] = conf3
            with open_dict(callback):
                callback.cfg = conf3
            continue
        elif callback['_target_'] == 'sunscc.callback.McIntoshClassificationConfusionMatrixCallback':
            with open_dict(callback):
                callback['output_dir'] = str(run_dir / outpath)
            
        callback = instantiate(callback, _recursive_=False)
        callback.conf = conf3  # FIXME : ugly hack
        callbacks.append(callback)

    trainer: pl.Trainer = instantiate(
        conf3.trainer,
        logger=conf3.logger,
        default_root_dir=".",
        callbacks=callbacks,
        _recursive_=True,
        _convert_="all",
    #     strategy="ddp",
        precision=16
    )
    trainer.tune(module, datamodule=datamodule)
    
    module.load_state_dict(torch.load(run_dir / "models/ENCODER_MLP1_MLP2_MLP3.pth"))

    et = datetime.now()
    print(f'INITIALIZED RUN IN {et-st}')
    print('STARTING EVALUATION')
    st = datetime.now()
    results = trainer.test(module, datamodule=datamodule)
    et = datetime.now()
    print(f'EVALUATED RUN IN {et-st}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_dir', type=str, help='run directory')
    parser.add_argument('--outpath', type=str, help='output directory')
    parser.add_argument('--use_npy', type=str, help='which dataset in npy file to use')
    args = parser.parse_args()
    main(args)