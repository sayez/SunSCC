

from email.mime import base
from tokenize import group
from sunscc.dataset.transform.pipelines import Compose
import collections
from functools import partial

from torch.utils.data import Dataset
import os
from hydra.utils import call, instantiate
from pathlib import Path
import skimage.io as io
import json

import numpy as np

import traceback
import time

from datetime import datetime, timedelta

from astropy.io import fits
from .utils import *


from copy import deepcopy 


def print_elapsed_time(st, msg):
        end_time = time.time()
        print(f'Elapsed time {msg}: {end_time - st}')



class DeepsunMaskedClassificationDataset(Dataset):

    def __init__(
        self, root_dir, partition, dtypes, classes, classification='Zurich' , transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        print(dtypes)

        self.transforms = transforms
        self.root_dir = Path(root_dir)

        self.main_dtype = dtypes[0]
        self.mask_dtype = dtypes[1]
        
        self.json_file = self.root_dir / (self.mask_dtype + '.json') 
        self.partition_dict = None

        with open(self.json_file, 'r') as f:
            self.partition_dict = json.load(f)[partition]

        assert len(dtypes) == 2, "dtypes > 2: DeepsunMaskedClassificationDataset can use only one type of mask."
        

        assert (classification == 'Zurich') or (classification == 'McIntosh')

        self.classification = classification

        self.classes_mapper = {c: i for i,c in enumerate(classes)}

        self.files = {}
        for i, bn in enumerate(sorted(list(self.partition_dict.keys()))):
            cur = {}
            image_basename = bn + '.FTS'
            image_filename = self.root_dir / self.main_dtype / image_basename

            mask_basename = bn + '.png'
            mask_filename = self.root_dir / self.mask_dtype / mask_basename

            cur["name"] = bn
            cur[self.main_dtype] = image_filename
            cur[self.mask_dtype] = mask_filename

            self.files[bn] = cur

        self.groups = {}
        for k,v in self.partition_dict.items():
            for i, g in enumerate(v['groups']):
                k2 = k+'_'+str(i)

                dataset_g = {   
                                "solar_angle": v['angle'],
                                "deltashapeX":v['deltashapeX'],
                                "deltashapeY":v['deltashapeY'],
                                "group_info": g,
                            }

                if dataset_g['group_info'][self.classification] in classes:
                    self.groups[k2] = dataset_g

        self.dataset_length = len(list(self.groups.keys()))



    def __len__(self) -> int:
        
        return self.dataset_length
    
    
    def __getitem__(self, index: int, do_transform=True):

        sample = {} # dictionnary with 'image', 'class', 'angular_excentricity', 'centroid_lat'

        # basename = self.files[index]["name"]
        k = sorted(list(self.groups.keys()))[index]
        basename = k.split('_')[0]

        # image_out_dict = self.partition_dict[basename]
        group_dict = self.groups[k]

        img_name = self.files[basename][self.main_dtype] # path of FITS file
        mask_name = self.files[basename][self.mask_dtype]

        sample['image'] = (io.imread(img_name)).astype(float) 
        sample['mask'] = io.imread(mask_name).astype(float)

        sample['solar_angle'] = group_dict['solar_angle']
        sample['deltashapeX'] = group_dict['deltashapeX']
        sample['deltashapeY'] = group_dict['deltashapeY']
        
        sample['angular_excentricity'] = np.array([group_dict['group_info']["angular_excentricity_deg"]])
        sample['centroid_px'] = np.array(group_dict['group_info']["centroid_px"])
        sample['centroid_Lat'] = np.array([group_dict['group_info']["centroid_Lat"]])

        sample['class'] = np.array([self.classes_mapper[group_dict['group_info'][self.classification]]])

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)

        # print([ (key,type(sample[key])) for key in list(sample.keys())])

        return sample
        


class DeepsunMaskedClassificationDatasetV2(Dataset):

    def __init__(
        self, root_dir, partition, dtypes, classes, classification='Zurich' , transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        print(dtypes)

        self.transforms = transforms
        self.root_dir = Path(root_dir)

        self.main_dtype = dtypes[0]
        self.mask_dtype = dtypes[1]
        
        self.json_file = self.root_dir / (self.mask_dtype + '.json') 
        self.partition_dict = None

        with open(self.json_file, 'r') as f:
            self.partition_dict = json.load(f)[partition]

        assert len(dtypes) == 2, "dtypes > 2: DeepsunMaskedClassificationDataset can use only one type of mask."
        

        assert (classification == 'Zurich') or (classification == 'McIntosh')

        self.classification = classification

        self.classes_mapper = {c: i for i,c in enumerate(classes)}

        # print(classes)
        self.files = {}
        for i, bn in enumerate(sorted(list(self.partition_dict.keys()))):
            bn = bn.split('_')[0]
            # print(bn)
            cur = {}
            image_basename = bn + '.FTS'
            image_filename = self.root_dir / self.main_dtype / image_basename

            mask_basename = bn + '.png'
            mask_filename = self.root_dir / self.mask_dtype / mask_basename

            cur["name"] = bn
            cur[self.main_dtype] = image_filename
            cur[self.mask_dtype] = mask_filename

            self.files[bn] = cur

        # print(self.files)

        self.partition_dict

        self.groups = {}
        for k,v in self.partition_dict.items():

            if v[self.classification] in classes:
                    self.groups[k] = v

        self.dataset_length = len(list(self.groups.keys()))



    def __len__(self) -> int:
        
        return self.dataset_length
    
    def __getitem__(self, index: int, do_transform=True):

        st = time.time()

        sample = {} # dictionnary with 'image', 'class', 'angular_excentricity', 'centroid_lat'

        # basename = self.files[index]["name"]
        k = sorted(list(self.groups.keys()))[index]
        # print(k)
        basename = k.split('_')[0]


        # image_out_dict = self.partition_dict[basename]
        group_dict = self.groups[k]

        # print(group_dict)

        img_name = self.files[basename][self.main_dtype] # path of FITS file
        mask_name = self.files[basename][self.mask_dtype]

        # print(img_name)

        sample['image'] = (io.imread(img_name)).astype(float) 
        sample['mask'] = io.imread(mask_name).astype(float)

        sample['solar_angle'] = group_dict['angle']
        sample['deltashapeX'] = group_dict['deltashapeX']
        sample['deltashapeY'] = group_dict['deltashapeY']
        
        sample['angular_excentricity'] = np.array([group_dict["angular_excentricity_deg"]])
        sample['centroid_px'] = np.array(group_dict["centroid_px"])
        sample['centroid_Lat'] = np.array([group_dict["centroid_Lat"]])

        sample['class'] = np.array([self.classes_mapper[group_dict[self.classification]]])

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)


        et = time.time()
        # print(f'DeepsunMaskedClassificationDatasetV2 getitem time: {et-st} seconds')

        return sample


class ClassificationDatasetSuperclasses(Dataset):

    def __init__(
        self, root_dir, partition, dtypes, classes,
        first_classes, second_classes, third_classes, 
        json_file, classification='SuperClass' , transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        print(dtypes)

        self.transforms = transforms
        self.root_dir = Path(root_dir)

        self.main_dtype = dtypes[0]
        self.mask_dtype = dtypes[1]

        self.json_file = self.root_dir / json_file 
        
        self.partition_dict = None

        self.c1_mapper = {c: i for i,c in enumerate(first_classes)}
        self.c2_mapper = {c: i for i,c in enumerate(second_classes)}
        self.c3_mapper = {c: i for i,c in enumerate(third_classes)}


        with open(self.json_file, 'r') as f:
            self.partition_dict = json.load(f)[partition]

        
        assert (classification == 'Zurich') or (classification == 'McIntosh') or (classification == 'SuperClass')

        self.classification = classification

        self.FirstClass_mapper = {c: i for i,c in enumerate(first_classes)}
        self.SecondClass_mapper = {c: i for i,c in enumerate(second_classes)}
        self.ThirdClass_mapper = {c: i for i,c in enumerate(third_classes)}

        # print(classes)
        self.files = {}
        for i, bn in enumerate(sorted(list(self.partition_dict.keys()))):
            bn = bn.split('_')[0]
            # print(bn)
            cur = {}
            image_basename = bn + '.FTS'
            image_filename = self.root_dir / self.main_dtype / image_basename

            sun_mask_filename = self.root_dir / 'sun_mask' / (bn + '.png')


            mask_basename = bn + '.png'
            mask_filename = self.root_dir / self.mask_dtype / mask_basename

            conf_map_basename = bn + '_proba_map.npy'
            conf_map_filename = self.root_dir / self.mask_dtype / conf_map_basename

            cur["name"] = bn
            cur[self.main_dtype] = image_filename
            cur[self.mask_dtype] = mask_filename
            cur[self.mask_dtype+"_conf_map"] = conf_map_filename
            cur["sun_mask"] = sun_mask_filename

            self.files[bn] = cur

        # print(self.files)

        self.partition_dict

        # print(list(self.partition_dict.values())[0])

        self.groups = {}
        for k,v in self.partition_dict.items():

            if v[self.classification]["1"] in classes:
                    self.groups[k] = v
            else:
                # print(v[self.classification]["1"])
                pass

      

        self.dataset_length = len(list(self.groups.keys()))



    def __len__(self) -> int:
        
        return self.dataset_length
    
    def __getitem__(self, index: int, do_transform=True):

        # st = time.time()

        sample = {} # dictionnary with 'image', 'class', 'angular_excentricity', 'centroid_lat'

        # basename = self.files[index]["name"]
        k = sorted(list(self.groups.keys()))[index]
        # print(k)
        basename = k.split('_')[0]

        group_dict = self.groups[k]

        img_name = self.files[basename][self.main_dtype] # path of FITS file
        mask_name = self.files[basename][self.mask_dtype]
        conf_map_name = self.files[basename][self.mask_dtype+"_conf_map"]

        # print(img_name)
        # st =  time.time()
        hdulst:fits.HDUList = fits.open(img_name)
        image = hdulst[0]
        header = image.header
        center = np.array(image.shape)//2
        radius = header['SOLAR_R']
        # print_elapsed_time(st, 'fits.open')
        
        # st = time.time()
        sample['solar_disk'] = io.imread(self.files[basename]["sun_mask"])
        # print_elapsed_time(st, 'load sun mask')



        # st = time.time()
        sample['excentricity_map'] = create_excentricity_map(sample['solar_disk'], radius, value_outside=-1)
        # print_elapsed_time(st, 'create_excentricity_map')

        # st = time.time()
        sample['mask'] = io.imread(mask_name)#.astype(float)
        sample['confidence_map'] = np.load(conf_map_name)
        # print_elapsed_time(st, 'load mask and conf map')

        # st = time.time()
        sample['image'] = (image.data).astype(float)

        sample['members'] = np.array(group_dict['members']) if 'members' in group_dict else np.array([0])
        sample['members_mean_px'] = np.array(group_dict['members_mean_px']) if 'members_mean_px' in group_dict else np.array([0])

        sample['name'] = basename

        sample['solar_angle'] = group_dict['angle']
        sample['deltashapeX'] = group_dict['deltashapeX']
        sample['deltashapeY'] = group_dict['deltashapeY']
        
        sample['angular_excentricity'] = np.array([group_dict["angular_excentricity_deg"]])
        sample['centroid_px'] = np.array(group_dict["centroid_px"])
        # print(sample['centroid_px'])
        sample['centroid_Lat'] = np.array([group_dict["centroid_Lat"]])

        # sample['class'] = np.array([self.classes_mapper[group_dict[self.classification]]])
        sample['class1'] = np.array([self.FirstClass_mapper[group_dict[self.classification]['1']]])
        sample['class2'] = np.array([self.SecondClass_mapper[group_dict[self.classification]['2']]])
        sample['class3'] = np.array([self.ThirdClass_mapper[group_dict[self.classification]['3']]])
        # print_elapsed_time(st, 'remaining operations')

        flip_time = "2003-03-08T00:00:00"
        date = whitelight_to_datetime(basename)
        datetime_str = datetime_to_db_string(date).replace(' ', 'T')
        # print(datetime_str)
        should_flip = (datetime.fromisoformat(datetime_str) - datetime.fromisoformat(flip_time)) < timedelta(0)
        sample['should_flip'] = should_flip

        if should_flip:
            sample['image'] = np.flip(sample['image'],axis=0)
            sample['solar_disk'] = np.flip(sample['solar_disk'],axis=0)
            sample['mask'] = np.flip(sample['mask'],axis=0)
            sample['confidence_map'] = np.flip(sample['confidence_map'],axis=0)
            sample['excentricity_map'] = np.flip(sample['excentricity_map'],axis=0)
    

        # print("transforms")
        # st = time.time()
        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)
        # print_elapsed_time(st, 'transforms')
        # print([ (key,type(sample[key])) for key in list(sample.keys())])

        # et = time.time()
        # print(f'DeepsunMaskedClassificationDatasetV2 getitem time: {et-st} seconds')
        # print()
        return sample



class ClassificationDatasetSuperclasses_fast(Dataset):

    def __init__(
        self, root_dir, partition, dtypes, classes,
        first_classes, second_classes, third_classes, 
        dataset_file, classification='SuperClass' , transforms=None) -> None:
        super().__init__()
        if isinstance(transforms, collections.abc.Mapping):
        # if isinstance(transforms, collections.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.abc.Sequence):
        # elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)

        self.transforms = transforms
        self.root_dir = Path(root_dir)

        self.main_dtype = dtypes[0]
        self.mask_dtype = dtypes[1]

        self.disk_file = self.root_dir / dataset_file.replace('.', '_'+partition+'.')
        
        self.partition_dict = None

        self.c1_mapper = {c: i for i,c in enumerate(first_classes)}
        self.c2_mapper = {c: i for i,c in enumerate(second_classes)}
        self.c3_mapper = {c: i for i,c in enumerate(third_classes)}
        
        st = time.time()
        print(f'Loading {partition} npy dataset')
        dataset = np.load(self.disk_file, allow_pickle=True).item()
        print_elapsed_time(st, 'Loading npy dataset')

        self.partition_dict = dataset

        self.classification = classification

        self.FirstClass_mapper = {c: i for i,c in enumerate(first_classes)}
        self.SecondClass_mapper = {c: i for i,c in enumerate(second_classes)}
        self.ThirdClass_mapper = {c: i for i,c in enumerate(third_classes)}

        # print(self.files)

        # print(list(self.partition_dict.values())[0])
        self.class_distrib = {
            'class1': {i:0 for i in range(len(first_classes))},
            'class2': {i:0 for i in range(len(second_classes))},
            'class3': {i:0 for i in range(len(third_classes))}
        }

        self.groups = {}
        for k,v in self.partition_dict.items():

            # if v["class1"] in classes:
            if ((v["class1"] in first_classes) and 
                (v["class2"] in second_classes) and 
                (v["class3"] in third_classes)):
                    self.groups[k] = v
                    self.class_distrib['class1'][self.FirstClass_mapper[v["class1"]]] += 1
                    self.class_distrib['class2'][self.SecondClass_mapper[v["class2"]]] += 1
                    self.class_distrib['class3'][self.ThirdClass_mapper[v["class3"]]] += 1
            else:
                # print(v[self.classification]["1"])
                pass

        print(self.class_distrib)

        self.dataset_length = len(list(self.groups.keys()))


    def __len__(self) -> int:
        
        return self.dataset_length
    
    def __getitem__(self, index: int, do_transform=True):

        idx = list(self.groups.keys())[index]
        
        sample = deepcopy(self.groups[idx])

        sample['image'][sample['excentricity_map'] < 0] = 0
        sample['mask'][sample['excentricity_map'] < 0] = 0
        sample['group_mask'][sample['excentricity_map'] < 0] = 0
        sample['solar_disk'][sample['excentricity_map'] < 0] = 0
        sample['confidence_map'][sample['excentricity_map'] < 0] = 0
        sample['group_confidence_map'][sample['excentricity_map'] < 0] = 0
        sample['group_confidence_map'][sample['excentricity_map'] > 0.95] = 0
        
        sample['excentricity_map'][sample['excentricity_map'] < 0] = 0

        sample.pop('should_flip', None)
        sample.pop('members_mean_px', None)
        sample.pop('centroid_px', None)
        sample.pop('name', None)
        
        sample['class1'] = np.array([self.FirstClass_mapper[sample['class1']]])
        sample['class2'] = np.array([self.SecondClass_mapper[sample['class2']]])
        sample['class3'] = np.array([self.ThirdClass_mapper[sample['class3']]])

        # st = time.time()
        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)
        # print_elapsed_time(st, 'transform')

        return sample

        