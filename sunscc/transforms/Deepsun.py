"""Functional interface to several data augmentation functions."""
from configparser import Interpolation
from re import L
from sre_constants import IN_IGNORE
from tokenize import group
from kornia import center_crop
import torch

from copy import deepcopy
import cv2
import numpy as np
from skimage.measure import label, regionprops

from albumentations.core.transforms_interface import DualTransform, BasicTransform
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.geometric.rotate import Rotate, RandomRotate90, SafeRotate
from albumentations.augmentations.transforms import Flip
import albumentations.augmentations.crops.functional as F_crops
from albumentations.augmentations import functional as F_aug
from albumentations.augmentations.geometric import functional as F_geom

import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

from scipy.ndimage.interpolation import rotate

import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.measure import label, regionprops

import time

class DeepsunCropPatch(DualTransform):
    def __init__(self, always_apply=False, p=0.5, patch_side=512):
        super().__init__(always_apply, p)
        self.always_apply = always_apply
        self.p = p
        self.patch_side = patch_side

    def get_grid_size(self, image):
        side = image.shape[0] // self.patch_side
        return side*side

    def crop_patch(self, img, patch_index):
        grid_size = self.get_grid_size(img)
        i = patch_index // int(np.sqrt(grid_size))
        j = patch_index % int(np.sqrt(grid_size))
        return img.copy()[
                            i*self.patch_side: (i+1)*self.patch_side,
                            j*self.patch_side: (j+1)*self.patch_side
                        ]

    def __call__(self, *args, force_apply=False, **kwargs):
        # print("DeepsunCropPatch")

        processed_kwargs = deepcopy(kwargs)

        # get patch index
        assert 'TTA_idx' in kwargs, "TTA_idx not in kwargs"
        # sample_id is formatted as: "imageIdx_patchIdx_augIdx"
        sample_id = kwargs['sample_id']
        patch_idx = int(sample_id.split("_")[1])

        processed_kwargs['image'] = self.crop_patch(processed_kwargs['image'], patch_idx)
        processed_kwargs['segmentation'] = self.crop_patch(processed_kwargs['segmentation'], patch_idx)
        if 'solar_disk' in kwargs:
            processed_kwargs['solar_disk'] = self.crop_patch(processed_kwargs['solar_disk'], patch_idx)

        if 'aug_history' in kwargs:
            params = {'patch_idx': patch_idx, 'patch_side': self.patch_side,
                        "original_image_shape": kwargs['image'].shape,
            }
            processed_kwargs['aug_history'].append({'DeepsunCropPatch': params})

        kwargs = deepcopy(processed_kwargs)
        return kwargs


class DeepsunTTARotate90(DualTransform):
    def __init__(self, always_apply=False, p=1):
        self.always_apply = always_apply
        self.p = p           
        super().__init__(always_apply=always_apply, p=p)

    def __call__(self, *args, force_apply=False, **kwargs):
        processed_kwargs = deepcopy(kwargs)
        # for k in kwargs:
        #     print(k)
        # print("DeepsunTTARotate90")
        assert 'TTA_idx' in kwargs, "TTA_idx not in kwargs"
        # print(processed_kwargs['TTA_idx'])
        if 'TTA_idx' in kwargs:
            if processed_kwargs['TTA_idx'] == 0:
                # print("First occurence of image -> Skip DeepsunTTARotate90")
                pass
            else:
                # print("DeepsunTTARotate90")
                # print(processed_kwargs['image'].shape, processed_kwargs['segmentation'].shape)
                processed_kwargs['image'] = np.rot90(processed_kwargs['image'].copy(), processed_kwargs['TTA_idx'])
                processed_kwargs['segmentation'] = np.rot90(processed_kwargs['segmentation'].copy(), processed_kwargs['TTA_idx'])
                # print(processed_kwargs['image'].shape, processed_kwargs['segmentation'].shape)
                
                if 'solar_disk' in kwargs:
                    processed_kwargs['solar_disk'] = np.rot90(processed_kwargs['solar_disk'].copy(), processed_kwargs['TTA_idx'])

            if 'aug_history' in kwargs:

                params = {'factor': processed_kwargs['TTA_idx']}
                # print("aug_history")
                processed_kwargs['aug_history'].append({'DeepsunTTARotate90': params})

        kwargs = deepcopy(processed_kwargs)
        return kwargs

class DeepsunTTAFlip(DualTransform):
    def __init__(self, always_apply=False, p=1):
        self.always_apply = always_apply
        self.p = p           
        super().__init__(always_apply=always_apply, p=p)

    def __call__(self, *args, force_apply=False, **kwargs):
        processed_kwargs = deepcopy(kwargs)
        # for k in kwargs:
        #     print(k)
        # print("DeepsunTTAFlip")
        assert 'TTA_idx' in kwargs, "TTA_idx not in kwargs"
        # print(processed_kwargs['TTA_idx'])
        if 'TTA_idx' in kwargs:
            if processed_kwargs['TTA_idx'] == 0:
                # print("First occurence of image -> Skip DeepsunTTAFlip")
                pass
            else:
                # print("DeepsunTTAFlip")
                # print(processed_kwargs['image'].shape, processed_kwargs['segmentation'].shape)
                processed_kwargs['image'] = np.flip(processed_kwargs['image'].copy(), processed_kwargs['TTA_idx'])
                processed_kwargs['segmentation'] = np.flip(processed_kwargs['segmentation'].copy(), processed_kwargs['TTA_idx'])
                # print(processed_kwargs['image'].shape, processed_kwargs['segmentation'].shape)
                
                if 'solar_disk' in kwargs:
                    processed_kwargs['solar_disk'] = np.flip(processed_kwargs['solar_disk'].copy(), processed_kwargs['TTA_idx'])

            if 'aug_history' in kwargs:
                # print("aug_history")
                processed_kwargs['aug_history'].append({'DeepsunTTAFlip': processed_kwargs['TTA_idx']})

        kwargs = deepcopy(processed_kwargs)
        return kwargs

    


class DeepsunRandomShiftScaleRotate(ShiftScaleRotate):
    def __init__(self, shift_limit=[-0.0625,0.0625], scale_limit=[1.,1.], rotate_limit=[-45,45], 
                        interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=-1, 
                        always_apply=False, p=1):
        super().__init__(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
                         interpolation=interpolation, border_mode=border_mode, value=value, mask_value=mask_value,
                         always_apply=always_apply, p=p)
        self.params = {}
        
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

        self.always_apply = always_apply
        self.p = p
        # print('init ShiftScaleRotate')

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        
        if 'solar_disk' in kwargs:
            mod_kwargs['masks'] = [ kwargs['segmentation'], kwargs['solar_disk'].astype(np.uint8)]
        else:
            mod_kwargs['mask'] = kwargs['segmentation']
        
        del mod_kwargs['segmentation']

        if ('TTA_idx' in mod_kwargs) and (mod_kwargs["TTA_idx"] == 0 ):
            # The image should not be modified during first passage through network at test time
            # print("First occurence of image -> Skip ShiftScaleRotate")
            processed_kwargs = mod_kwargs
        else:
            processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']

        if 'aug_history' in kwargs:
            no_modif = {'angle': 0., 'scale': 1.0, 'dx': 0., 'dy':0.,
                            'interpolation': 1, 'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]}
            if ((len(processed_kwargs['aug_history']) == 0) or # transform was not applied (probability self.p)
                     ('DeepsunRandomShiftScaleRotate' not in processed_kwargs['aug_history'][-1]) or #
                     ( ('TTA_idx' in mod_kwargs) and (mod_kwargs["TTA_idx"] == 0 )) # first occurence of image in TTA should not apply transform
                ):
                processed_kwargs['aug_history'].append({'DeepsunRandomShiftScaleRotate': no_modif})

            
            kwargs['aug_history'] = processed_kwargs['aug_history']
            # print(f"DeepsunRandomShiftScaleRotate args:,{kwargs['aug_history'][-1]} \n")

        if 'solar_disk' in kwargs:
            kwargs['solar_disk'] = processed_kwargs['masks'][-1].astype(np.bool8)
            # kwargs['segmentation'] = processed_kwargs['masks'][0]* processed_kwargs['masks'][1]
            kwargs['segmentation'] = processed_kwargs['masks'][0]
            kwargs['segmentation'][processed_kwargs['masks'][1]==0] = self.mask_value
            kwargs['segmentation'][processed_kwargs['image']==0] = self.mask_value
        else:
            kwargs['segmentation'] = processed_kwargs['mask']


        return kwargs

    def update_params(self, params, **kwargs):
        tmp = super().update_params(params, **kwargs)
        # print(f'update_param {kwargs["sample_id"]} :{tmp}')

        if 'aug_history' in kwargs:
            kwargs['aug_history'].append({'DeepsunRandomShiftScaleRotate':tmp})
        # self.params[kwargs['sample_id']] = tmp
        return tmp

    def apply_to_masks(self, masks, **params):
        # print("DeepsunRandomShiftScaleRotate",params)
        # self.params[params['sample_id']] = {**params}
        return [self.apply_to_mask(mask, **params) for mask in masks]



class DeepsunClassificationRandomFlip(Flip):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['mask'] = kwargs['mask']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['mask'] = processed_kwargs['mask']
        return kwargs


class DeepsunClassificationRandomRotate(SafeRotate):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, 
                        value=None, mask_value=None, always_apply=False, p=0.5):
        super().__init__(limit, interpolation, border_mode, value, mask_value, always_apply, p)

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['mask'] = kwargs['mask']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['mask'] = processed_kwargs['mask']
        return kwargs

class DeepsunRandomRotate(SafeRotate):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, 
                        # border_mode=cv2.BORDER_REFLECT_101, 
                        # border_mode=cv2.BORDER_REPLICATE, 
                        # border_mode=cv2.BORDER_WRAP, 
                        border_mode=cv2.BORDER_REFLECT, 
                        value=None, mask_value=None, always_apply=False, p=1.0):
        super().__init__(limit, interpolation, border_mode, value, mask_value, always_apply, p)
        self.params = {}
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['masks']

        self.params = self.get_params()

        if 'solar_disk' in kwargs:
            kwargs['solar_disk'] = F_geom.safe_rotate(kwargs['solar_disk'].astype(np.uint8), 
                                                        self.params['angle'], 
                                                        interpolation= self.interpolation,                                                      
                                                        border_mode = self.border_mode                                                      
                                                    )
            kwargs['solar_disk'] = kwargs['solar_disk'].astype(np.bool8)

        if 'aug_history' in kwargs:
            kwargs['aug_history'].append({'DeepsunRandomRotate':self.params.copy()})
            print(f"RandomRotate: { self.params}")

        return kwargs  

    def reverse_transform(self):
        print(f'I Should Reverse RandomRotate {self.get_params()}')
        pass


class DeepsunRandomFlip(Flip):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.params = {}

    def __call__(self, *args, force_apply=False, **kwargs):
        # print(f'DeepsunRandomFlip: {kwargs["name"]}')
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']       
        # mod_kwargs['mask'] = kwargs['segmentation']
        if 'solar_disk' in kwargs:
            # print(f'DeepsunRandomFlip: image {mod_kwargs["image"].shape}')
            # print(f'DeepsunRandomFlip: solar_disk {kwargs["solar_disk"].shape}')
            # mod_kwargs['masks'] = [ kwargs['segmentation'], kwargs['solar_disk'].astype(np.uint8)]
            # mod_kwargs['masks'] = [ kwargs['segmentation'], kwargs['solar_disk'].astype(np.uint8)[None,:,:]]
            # stack solar disk to the end of the kwargs['segmentation'] array
            mod_kwargs['masks'] = np.concatenate([kwargs['segmentation'], kwargs['solar_disk'].astype(np.uint8)[None,:,:]], axis=0)
            # print(f'DeepsunRandomFlip: masks {mod_kwargs["masks"].shape}')
        else:
            # print('DeepsunRandomFlip: NO solar_disk')
            mod_kwargs['mask'] = kwargs['segmentation']
        

        del mod_kwargs['segmentation']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        if 'solar_disk' in kwargs:
            kwargs['solar_disk'] = processed_kwargs['masks'][-1].astype(np.bool8)
            # kwargs['solar_disk'] = processed_kwargs['masks'][-1][0].astype(np.bool8)
            kwargs['segmentation'] = processed_kwargs['masks'][:-1]
            # kwargs['segmentation'] = processed_kwargs['masks'][0]
        else:
            kwargs['segmentation'] = processed_kwargs['mask']
        
        if 'aug_history' in kwargs:
            kwargs['aug_history'].append({'DeepsunRandomFlip':self.params.copy()})
            # print(f"RandomFlip: { self.params}")
        return kwargs

    def apply(self, img, d=0, **params):
        self.params = {'d':d}
        # print(img.shape)
        return F_aug.random_flip(img, d)

    
    def apply_to_masks(self, masks, **params):
        # print('masks')
        self.params = {**params}
        tmp = []
        # for i,mask in enumerate(masks):
        #     print(f'mask [{i}]: {mask.shape}')
        #     tmp.append(self.apply_to_mask(mask, **params))
        #     print(f'mask [{i}]: fini')
        # return tmp
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def reverse_transform(self):
        print(f'I Should Reverse Randomflip {self.get_params()}')
        pass

class DeepsunCropNonEmptyMaskIfExists_SingleMask(CropNonEmptyMaskIfExists):
    def __init__(self, height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        
        super().__init__(height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0)
        # print('Init')
        
    def __call__(self, *args, force_apply=False, **kwargs):
        # print(kwargs)
        mod_kwargs = deepcopy(kwargs)
        # mod_kwargs['image'] = kwargs['sample']['image']
        # mod_kwargs['mask'] = kwargs['segmbyyentation']
        mod_kwargs['mask'] = kwargs['segmentation']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)
#         return processed_kwargs
        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['mask']

        return kwargs

    def apply_to_mask(self, img, **params):        
        X =  super().apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})
        # print("CROPNONEMPTY: ", X.shape)
        return X


class DeepsunCropNonEmptyMaskIfExists(CropNonEmptyMaskIfExists):
    def __init__(self, height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        super().__init__(height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0)
        self.params = {}
            
    def __call__(self, *args, force_apply=False, **kwargs):       
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        # print(mod_kwargs.keys())
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['masks']

        
        # print(f'CropNonEmptyMaskIfExists: { self.params }')

        if 'solar_disk' in kwargs:
            kwargs['solar_disk'] = F_crops.crop(kwargs['solar_disk'],
                                    x_min=self.params['x_min'], 
                                    x_max=self.params['x_max'], 
                                    y_min=self.params['y_min'], 
                                    y_max=self.params['y_max'])
        
        if 'aug_history' in kwargs:
            kwargs['aug_history'].append({'DeepsunCropNonEmptyMaskIfExists':self.params.copy()})


        return kwargs
    
    def reverse_transform(augmented):
        pass


    def apply_to_mask(self, img, **params): 
        self.params = params  
        X =  super().apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})
        return X
    
    def apply_to_masks(self, masks, **params):     
        X =  super().apply_to_masks(masks, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})
        return X
    
    def update_params(self, params, **kwargs):
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = deepcopy(kwargs["masks"])
            mask = self._preprocess_mask(masks[0].astype(np.uint8))
            for m in masks[1:]:
                mask |= self._preprocess_mask(m.astype(np.uint8))
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")
        
        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, self.width - 1)
            y_min = y - random.randint(0, self.height - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height
        # print({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params



class DeepsunReduceContrast_TTA(BasicTransform):
    def __init__(self, factor_ranges, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.factor_ranges = factor_ranges
        self.params = {}
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):      
        
        img = kwargs['image'].copy()
        disk_mask = kwargs['solar_disk']
        inside_disk = np.logical_and(disk_mask.astype(np.bool8), img>0)

        tta_idx = kwargs['TTA_idx']

        cur_factor = self.factor_ranges[tta_idx]

        mean = np.mean(img[inside_disk])
        min_inside = np.min(img[inside_disk])
        max_inside = np.max(img[inside_disk])
        delta = max_inside - min_inside
    
        normalized = img.copy()
        normalized[inside_disk] = cv2.normalize(img,None,0,cur_factor*delta,cv2.NORM_MINMAX)[inside_disk]
        reduced_mean = normalized[inside_disk].mean()
        normalized2 = normalized.copy()
        normalized2[inside_disk] += mean - reduced_mean

        kwargs['image'] = normalized2

        self.params = {"orig_mean": mean, "reduced_mean": reduced_mean, 
                        "delta": delta , "factor": cur_factor}

        if 'aug_history' in kwargs:
            # print('should add aug_history')
            kwargs['aug_history'].append({'DeepsunReduceContrast_TTA':self.params.copy()})


        # axis_side = 2
        # fig,ax = plt.subplots(nrows=1, ncols=5, 
        #                 figsize=(axis_side*4,axis_side*1),
        #                 dpi=100)

        # ax[0].imshow(img,cmap='gray') 
        # ax[0].set_title(f'Original') 
        # ax[1].hist(img.flatten(),bins=255) 

        # ax[2].imshow(disk_mask)

        # ax[3].imshow(normalized2,cmap= cm.Greys_r, vmin=0, vmax=img.max()) 
        # ax[3].set_title(f'{cur_factor} contrast')
        # ax[4].hist(normalized2.flatten(),bins=255)

        # plt.savefig(f'./test_segmentation_{self.index}.png', dpi=150)
        # plt.close()
        # self.index+=1
        
        return kwargs

    def reverse_transform(self,params):
        pass

class DeepsunRandomReduceContrast(BasicTransform):
    def __init__(self, factor_ranges, factor_probabilities, always_apply=False, p=1.):
        super().__init__(always_apply, p)
        self.factor_ranges = factor_ranges
        self.factor_probabilities = factor_probabilities
        
        np.testing.assert_almost_equal(np.array(self.factor_probabilities).sum(), 1.0, decimal=7,
                                        err_msg='Probabilities should add up to 1', verbose=False)
        
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):      
        
        img = kwargs['image'].copy()
        disk_mask = kwargs['solar_disk']
        inside_disk = disk_mask.astype(np.bool8)

        idx = int(np.random.choice(np.arange(0, len(self.factor_ranges)), 
                                p=self.factor_probabilities))

        cur_factor = self.factor_ranges[idx]

        np.seterr(all='raise') 

        try:
            # print(f"Avant {self.index}")
            mean = np.mean(img[inside_disk])
            # print(f"Apres {self.index}")
            min_inside = np.min(img[inside_disk])
            max_inside = np.max(img[inside_disk])
            delta = max_inside - min_inside
        
            normalized = img.copy()
            normalized[inside_disk] = cv2.normalize(img,None,0,cur_factor*delta,cv2.NORM_MINMAX)[inside_disk]
            reduced_mean = normalized[inside_disk].mean()
            normalized2 = normalized.copy()
            normalized2[inside_disk] += mean - reduced_mean

            kwargs['image'] = normalized2
        except FloatingPointError:
            # simply means that the patch does not show any part of the sun
            # So we do nothing

            # print("FloatingPointError")
            # print(kwargs['sample_id2'])

            # fig,ax = plt.subplots(nrows=1, ncols=3,figsize=(6,2),dpi=100)
            # ax[0].imshow(kwargs['image'],cmap='gray')
            # ax[1].imshow(kwargs['solar_disk'])
            # plt.show()

            # raise FloatingPointError
            pass


        # axis_side = 2
        # fig,ax = plt.subplots(nrows=1, ncols=5, 
        #                 figsize=(axis_side*4,axis_side*1),
        #                 dpi=100)

        # ax[0].imshow(img,cmap='gray') 
        # ax[0].set_title(f'Original') 
        # ax[1].hist(img.flatten(),bins=255) 

        # ax[2].imshow(disk_mask)

        # ax[3].imshow(normalized2,cmap= cm.Greys_r, vmin=0, vmax=img.max()) 
        # ax[3].set_title(f'{cur_factor} contrast')
        # ax[4].hist(normalized2.flatten(),bins=255)

        # plt.savefig(f'./test_segmentation_{self.index}.png', dpi=150)
        # plt.close()

        self.index+=1
        
        return kwargs


class DeepsunRandomMaskSelector(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(always_apply=False, p=1.0)
        
    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunMaskMerger')
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        # print(mod_kwargs.keys())
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)
        
        # print("processed", processed_kwargs.keys())
        # print(processed_kwargs['masks'])
        processed_kwargs['mask'] = processed_kwargs['masks']
        del processed_kwargs['masks']

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['mask']
        
        return kwargs
    
    def apply(self, img, **params):
        return img
    
    
    def apply_to_masks(self, masks, **params):  
        rnd_idx =  random.randint(0,len(masks)-1)
        # print(rnd_idx) 
        out_mask = masks[rnd_idx].copy()
       
        return out_mask

class DeepsunMaskMerger(DualTransform):
    def __init__(self, p_add = 0.5,  always_apply=False, p=1.0):     
        super().__init__(always_apply=False, p=1.0)
        self.p_add = p_add
        
    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunMaskMerger')
        mod_kwargs = deepcopy(kwargs)
        mod_kwargs['masks'] = kwargs['segmentation']
        del mod_kwargs['segmentation']
        # print(mod_kwargs.keys())
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)
        
        # print("processed", processed_kwargs.keys())
        # print(processed_kwargs['masks'])
        processed_kwargs['mask'] = processed_kwargs['masks']
        del processed_kwargs['masks']

        kwargs['image'] = processed_kwargs['image']
        kwargs['segmentation'] = processed_kwargs['mask']
        
        return kwargs

    
    def get_transform_init_args_names(self):
        return ("p_add",)
    
    def apply(self, img, **params):
        return img
    
    
    def apply_to_masks(self, masks, **params):   
        # we consider that the masks are ordered by higher threshold values first
        # -> more small sunspots (maybe fale positives), and large sunspots may merge with close others (LOW)
        # -> less small sunspots (more false negatives), but close sunspots are not merged
        # print('Masks merger')
        
        ####### FIND REGION PROPERTIES IN MASKS
        High_fg_bg = masks[0].copy()
        High_fg_bg[High_fg_bg>0] = 1    
        
        Low_fg_bg = masks[-1].copy()
        Low_fg_bg[Low_fg_bg>0] = 1

        label_low = label(Low_fg_bg)    
        props_labels_low = regionprops(label_low)
        
        
        ####### MERGE THE MASKS
        # 1) Take the High Threshold mask         
        out_mask = masks[0].copy()
        
        # 2) Add the small sunspots in Low threshold mask that do not intersect with other sunspots
        #    with a random condition
        for propLow in props_labels_low:
            bbox= propLow.bbox
            submask = propLow.image 
        
            cur_m_fg_bg = np.zeros_like(out_mask)
            cur_m_fg_bg[bbox[0]:bbox[2], bbox[1]:bbox[3]] = submask
            cur_m = cur_m_fg_bg*masks[-1].copy()
            
            intersection = cur_m * out_mask
            if (np.sum(intersection) == 0)  and  (random.random() < self.p_add):
                out_mask += cur_m
        
        return out_mask


def rotate_CV_bound(image, angle, interpolation):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),flags=interpolation)



class DeepsunRotateAndCropAroundGroup(DualTransform):

    def __init__(self, standard_height, standard_width,  always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.standard_height = standard_height
        self.standard_width = standard_width

        self.index = 0

    def expand_small_spots(self, msk):
        out_msk = msk.copy()
        label_img = label(out_msk)
        regions = regionprops(label_img)
        
        for r in regions:
            if r.area == 1:
                coords = r.coords[0]
                # print(coords)
                out_msk[coords[0]-1:coords[0]+1,coords[1]-1:coords[1]+1] = msk[coords[0],coords[1]]
                
        return out_msk

    def padder(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')

        st = time.time()

        img = kwargs['image']
        msk = kwargs['mask']
        disk = kwargs['solar_disk'].astype(np.uint8)

        # print(disk.dtype)

        try:
            # Make sure that all the single-pixels spots in mask are expanded to 3x3
            msk = self.expand_small_spots(msk)
            # 1) Correct solar Angle ->  Rotate image + Zoom In
            angle = kwargs['solar_angle']
            deltashapeX = kwargs['deltashapeX']
            deltashapeY = kwargs['deltashapeY']
            # print('solar_angle', angle)

            # rot_img = rotate(img, angle=angle, reshape=True)
            # rot_msk = rotate(msk, angle=angle, reshape=True)
            #rot_img = rotate_CV_bound(img, angle=angle, interpolation=cv2.INTER_LINEAR)
            #rot_msk = rotate_CV_bound(msk, angle=angle, interpolation=cv2.INTER_LINEAR)
            rot_img = rotate_CV_bound(img, angle=angle, interpolation=cv2.INTER_NEAREST)
            rot_msk = rotate_CV_bound(msk, angle=angle, interpolation=cv2.INTER_NEAREST)
            rot_disk = rotate_CV_bound(disk, angle=angle, interpolation=cv2.INTER_NEAREST)

            rot_img_zoom = rot_img[deltashapeX//2:rot_img.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_img.shape[1]-deltashapeY//2] 
            rot_msk_zoom = rot_msk[deltashapeX//2:rot_msk.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_msk.shape[1]-deltashapeY//2] 
            rot_disk_zoom = rot_disk[deltashapeX//2:rot_disk.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_disk.shape[1]-deltashapeY//2] 

            # print(rot_img_zoom.shape, rot_msk_zoom.shape)
            assert rot_img_zoom.shape == rot_msk_zoom.shape

            # 2) Crop around group
            group_centroid = np.array(kwargs['centroid_px'])
            # print(group_centroid)

            # minX = int(group_centroid[0])-self.standard_width//2
            # maxX = int(group_centroid[0])+self.standard_width//2
            # minY = int(group_centroid[1])-self.standard_height//2
            # maxY = int(group_centroid[1])+self.standard_height//2

            # img_group_crop = rot_img_zoom[minX:maxX,minY:maxY]
            # msk_group_crop = rot_msk_zoom[minX:maxX,minY:maxY]

            minX = self.standard_height + (int(group_centroid[1])-self.standard_width//2)
            maxX = self.standard_height + (int(group_centroid[1])+self.standard_width//2)
            minY = self.standard_height + (int(group_centroid[0])-self.standard_height//2)
            maxY = self.standard_height + (int(group_centroid[0])+self.standard_height//2)


            pad_rot_img_zoom = np.pad(rot_img_zoom, self.standard_height, self.padder, padder=0)
            pad_rot_msk_zoom = np.pad(rot_msk_zoom, self.standard_height, self.padder, padder=0)
            pad_rot_disk_zoom = np.pad(rot_disk_zoom, self.standard_height, self.padder, padder=0)

            img_group_crop = pad_rot_img_zoom[minX:maxX,minY:maxY]
            msk_group_crop = pad_rot_msk_zoom[minX:maxX,minY:maxY]
            disk_group_crop = pad_rot_disk_zoom[minX:maxX,minY:maxY]

            assert img_group_crop.shape == msk_group_crop.shape

        except AssertionError:
            pass
            # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
            # img = kwargs['image'].copy()
            # ax[0].set_title(kwargs["name"])
            # ax[1].set_title(kwargs["centroid_px"])
            # ax[0].imshow(img, interpolation=None, cmap='gray')
            # ax[0].imshow(msk, interpolation=None, alpha=0.5)
            # ax[1].imshow(rot_img, interpolation=None, cmap='gray')
            # ax[1].imshow(rot_msk, interpolation=None, alpha=0.5)
            # ax[2].imshow(img_group_crop, interpolation=None, cmap='gray')
            # ax[2].imshow(msk_group_crop, interpolation=None, alpha=0.5)
            
            # plt.show()
        
        # print(msk_group_crop.shape)

        kwargs.pop('solar_angle',None)
        kwargs.pop('deltashapeX',None)
        kwargs.pop('deltashapeY',None)
        # kwargs.pop('centroid_px',None)

        # As groups in a single batch can have different number of members, 
        # we need to remove the 'members' key from the kwargs so that they can be stacked.
        kwargs.pop('members',None)

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        # ax[0].imshow(img, interpolation=None, cmap='gray')
        # ax[0].imshow(msk, interpolation=None, alpha=0.5)
        # ax[1].imshow(rot_img_zoom, interpolation=None, cmap='gray')
        # ax[1].imshow(rot_msk_zoom, interpolation=None, alpha=0.5)
        # ax[2].imshow(img_group_crop, interpolation=None, cmap='gray')
        # ax[2].imshow(msk_group_crop, interpolation=None, alpha=0.5)
        # ax[1].scatter(group_centroid[0],group_centroid[1], c='r', s=1 )
        # plt.savefig(f'./test_classification_{self.index}.png', dpi=150)
        
        self.index+=1


        kwargs['image'] = img_group_crop.copy()
        kwargs['mask'] = msk_group_crop.copy()
        kwargs['solar_disk'] = disk_group_crop.copy()


        et = time.time()
        # print(f'DeepsunRotateAndCropAroundGroup time: {et-st} seconds')
        
        
        return kwargs

        
class Deepsun_Focus_Move(DualTransform):
    def __init__(self, standard_height=256, standard_width=256,
                        focus_on_group=True,
                        random_move=False, random_move_percent=0.1,  
                        always_apply=False, p=1.0) -> None:

        super().__init__(always_apply, p)

        self.standard_height = standard_height
        self.standard_width = standard_width
        
        self.focus_on_group = focus_on_group
        self.random_move = random_move
        self.random_move_percent = random_move_percent

    def get_bounding_box_around_group_with_padding(self, mask, offset):
        # Get the bounding box around non-zero pixels in mask
        x, y = np.nonzero(mask)
        # print(x, y)
        x1, x2 = (np.min(x), np.max(x)) if len(x) > 0 else (None, None)
        y1, y2 = (np.min(y), np.max(y)) if len(y) > 0 else (None, None)

        if (x1 is None) or (y1 is None):
            return 0, mask.shape[0]-1, 0, mask.shape[1]-1

        # Add padding
        x1 -= offset
        x2 += offset
        y1 -= offset
        y2 += offset


        # Make sure the bounding box is not outside the image
        x1 = max(x1, 0)
        x2 = min(x2, mask.shape[0])
        y1 = max(y1, 0)
        y2 = min(y2, mask.shape[1])

        return x1, x2, y1, y2

    def adapt_bbox_to_image_size(self, bbox, image_size):
        bbox_center = ((bbox[0] + bbox[1]) // 2, (bbox[2] + bbox[3]) // 2)
        bbox_size = (bbox[1] - bbox[0], bbox[3] - bbox[2])

        # if bbox is too small, expand it
        minimal_percentage = .4

        bbox_size = (max(bbox_size[0], image_size[0] * minimal_percentage),
                     max(bbox_size[1], image_size[1] * minimal_percentage))
        
        return (int(bbox_center[0] - bbox_size[0] // 2), int(bbox_center[0] + bbox_size[0] // 2),
                int(bbox_center[1] - bbox_size[1] // 2), int(bbox_center[1] + bbox_size[1] // 2))

    def crop_img(self, img, bbox):
        # Crop image
        x1, x2, y1, y2 = bbox
        img = img[x1:x2, y1:y2]
        return img
        
    def padding(self, array, xx, yy):
        """
        :param array: numpy array
        :param xx: desired height
        :param yy: desirex width
        :return: padded array
        """

        h = array.shape[0]
        w = array.shape[1]

        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        # print(a,aa,b,bb)

        a = max(a,0)
        b = max(b,0)
        aa = max(aa,0)
        bb = max(bb,0)


        # print('->',a,aa,b,bb)

        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

    def crop_and_pad(self, img, bbox, image_size):
        # print('crop_and_pad')
        # print(f'bbox: {bbox}, image_size: {image_size}, img.shape: {img.shape}')
        # Crop image
        img = self.crop_img(img, bbox)
        # Pad image
        img = self.padding(img, image_size[0], image_size[1])
        return img

    def data_aug_random_move(self, bbox, max_offset):
        '''
        Randomly move the bounding box
        param bbox: bounding box
        param max_offset: maximum offset in portion of the bbox size
        '''
        # Randomly move the bounding box
        x1, x2, y1, y2 = bbox
        horizontal_offset = (np.random.random(1) * 2*max_offset) - max_offset
        vertical_offset = (np.random.random(1) * 2*max_offset) - max_offset
        # print(f'horizontal_offset: {horizontal_offset}, vertical_offset: {vertical_offset}')

        x1 += int(horizontal_offset * (bbox[1] - bbox[0]))
        x2 += int(horizontal_offset * (bbox[1] - bbox[0]))
        y1 += int(vertical_offset * (bbox[3] - bbox[2]))
        y2 += int(vertical_offset * (bbox[3] - bbox[2]))
        
        return x1, x2, y1, y2
        
    def __call__(self, *args, force_apply=False, **kwargs):

        img_group_crop = kwargs['image'].copy()
        msk_group_crop = kwargs['mask'].copy()
        grp_msk_group_crop = kwargs['group_mask'].copy()
        disk_group_crop = kwargs['solar_disk'].copy()
        excentricity_group_crop = kwargs['excentricity_map'].copy()
        confidence_group_crop = kwargs['confidence_map'].copy()
        grp_confidence_group_crop = kwargs['group_confidence_map'].copy()
        
        shape  = img_group_crop.shape
        # minX, maxX, minY, maxY =  ((shape[0]//2)-self.standard_height//2, 
        #                             (shape[0]//2)+self.standard_height//2, 
        #                             (shape[1]//2)-self.standard_width//2, 
        #                             (shape[1]//2)+self.standard_width//2)

        # bbox format = x1, x2, y1, y2
        bbox = self.get_bounding_box_around_group_with_padding((grp_confidence_group_crop>0), 10)

        # print(bbox)
        # print(bbox[1]-bbox[0], bbox[3]-bbox[2])
        
        minX, maxX, minY, maxY =  (
                                    ((bbox[1]+bbox[0])//2)-(self.standard_height//2), 
                                    ((bbox[1]+bbox[0])//2)+(self.standard_height//2), 
                                    ((bbox[3]+bbox[2])//2)-(self.standard_width//2), 
                                    ((bbox[3]+bbox[2])//2)+(self.standard_width//2)
                                    
                                    )
        # print("new_shape minmax ",minX, maxX, minY, maxY)
        
            
        if self.focus_on_group:
            # focus on the group
            # print('focus on group')
            # Modify the bounding box if data augmentation is enabled
            if self.random_move:
                # print('random_move')
                bbox = self.data_aug_random_move(bbox, max_offset=self.random_move_percent)
                                    
                # Make sure the bounding box is not outside the image
                x1, x2, y1, y2 = bbox
                x1 = max(x1, 0)
                x2 = min(x2, self.standard_width)
                y1 = max(y1, 0)
                y2 = min(y2, self.standard_height)
                bbox = x1, x2, y1, y2
            else:
                # print('no random_move')
                pass
                
            bbox = self.adapt_bbox_to_image_size( bbox, (self.standard_height, self.standard_width))
            # print(bbox)
            img_group_crop = self.crop_and_pad(img_group_crop, bbox, (self.standard_height, self.standard_width))
            msk_group_crop = self.crop_and_pad(msk_group_crop, bbox, (self.standard_height, self.standard_width))
            grp_msk_group_crop = self.crop_and_pad(grp_msk_group_crop, bbox, (self.standard_height, self.standard_width))
            disk_group_crop = self.crop_and_pad(disk_group_crop, bbox, (self.standard_height, self.standard_width))
            excentricity_group_crop = self.crop_and_pad(excentricity_group_crop, bbox, (self.standard_height, self.standard_width))
            confidence_group_crop = self.crop_and_pad(confidence_group_crop, bbox, (self.standard_height, self.standard_width))
            grp_confidence_group_crop = self.crop_and_pad(grp_confidence_group_crop, bbox, (self.standard_height, self.standard_width))
        else:
            # print('NO focus on group')
            if self.random_move:
                # print('random_move')
                frac = np.max([(bbox[1]-bbox[0]) /self.standard_height, (bbox[3]-bbox[2]) /self.standard_width])
                frac = np.sqrt(frac)
                # print(minX, maxX, minY, maxY , frac, self.random_move_percent)
                bbox = self.data_aug_random_move([minX,maxX,minY,maxY], max_offset=self.random_move_percent*frac)
                # print('bbox',bbox)
                if not ((bbox[1] > shape[0]) or (bbox[3] > shape[1]) or (bbox[0] < 0) or (bbox[2] < 0)):
                    bbox = [bbox[0], bbox[0]+self.standard_height, bbox[2], bbox[3]]
                    # x1, x2, y1, y2 = bbox
                    # x1 = max(x1, 0)
                    # x2 = min(x2, self.standard_width)
                    # y1 = max(y1, 0)
                    # y2 = min(y2, self.standard_height)
                    minX,maxX,minY,maxY = bbox
                # print(minX, maxX, minY, maxY )
            else:
                # print('no random_move')
                pass
                
            img_group_crop = img_group_crop[minX:maxX,minY:maxY]
            msk_group_crop = msk_group_crop[minX:maxX,minY:maxY]
            grp_msk_group_crop = grp_msk_group_crop[minX:maxX,minY:maxY]
            disk_group_crop = disk_group_crop[minX:maxX,minY:maxY]
            excentricity_group_crop = excentricity_group_crop[minX:maxX,minY:maxY]
            confidence_group_crop = confidence_group_crop[minX:maxX,minY:maxY]
            grp_confidence_group_crop = grp_confidence_group_crop[minX:maxX,minY:maxY]

        # self.print_elapsed_time(st, 'focusMove Operations')
        # print('after focus_move call',img_group_crop.shape)

        if img_group_crop.shape != (self.standard_height, self.standard_width):
            img_group_crop = self.padding(img_group_crop, self.standard_height, self.standard_width)
            msk_group_crop = self.padding(msk_group_crop, self.standard_height, self.standard_width)
            grp_msk_group_crop = self.padding(grp_msk_group_crop, self.standard_height, self.standard_width)
            disk_group_crop = self.padding(disk_group_crop, self.standard_height, self.standard_width)
            excentricity_group_crop = self.padding(excentricity_group_crop, self.standard_height, self.standard_width)
            confidence_group_crop = self.padding(confidence_group_crop, self.standard_height, self.standard_width)
            grp_confidence_group_crop = self.padding(grp_confidence_group_crop, self.standard_height, self.standard_width)


        assert img_group_crop.shape == (self.standard_height, self.standard_width)
        assert msk_group_crop.shape == (self.standard_height, self.standard_width)
        assert grp_msk_group_crop.shape == (self.standard_height, self.standard_width)
        assert disk_group_crop.shape == (self.standard_height, self.standard_width)
        assert excentricity_group_crop.shape == (self.standard_height, self.standard_width)
        assert confidence_group_crop.shape == (self.standard_height, self.standard_width)
        assert grp_confidence_group_crop.shape == (self.standard_height, self.standard_width)



        kwargs['image'] = img_group_crop.copy()
        kwargs['mask'] = msk_group_crop.copy()
        kwargs['group_mask'] = grp_msk_group_crop.copy()
        kwargs['solar_disk'] = disk_group_crop.copy()
        kwargs['excentricity_map'] = excentricity_group_crop.copy()
        kwargs['confidence_map'] = confidence_group_crop.copy()
        kwargs['group_confidence_map'] = grp_confidence_group_crop.copy()
        
        return kwargs

class DeepsunClassifRandomFlip(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def __call__(self, *args, force_apply=False, **kwargs):
        
        img_group_crop = kwargs['image'].copy()
        msk_group_crop = kwargs['mask'].copy()
        grp_msk_group_crop = kwargs['group_mask'].copy()
        disk_group_crop = kwargs['solar_disk'].copy()
        excentricity_group_crop = kwargs['excentricity_map'].copy()
        confidence_group_crop = kwargs['confidence_map'].copy()
        grp_confidence_group_crop = kwargs['group_confidence_map'].copy()


        # generate parameters to flip horizontally and vertically
        hflip = np.random.choice([True, False])
        vflip = np.random.choice([True, False])

        # flip image
        if hflip:
            img_group_crop = np.fliplr(img_group_crop)
            msk_group_crop = np.fliplr(msk_group_crop)
            grp_msk_group_crop = np.fliplr(grp_msk_group_crop)
            disk_group_crop = np.fliplr(disk_group_crop)
            excentricity_group_crop = np.fliplr(excentricity_group_crop)
            confidence_group_crop = np.fliplr(confidence_group_crop)
            grp_confidence_group_crop = np.fliplr(grp_confidence_group_crop)
        
        if vflip:
            img_group_crop = np.flipud(img_group_crop)
            msk_group_crop = np.flipud(msk_group_crop)
            grp_msk_group_crop = np.flipud(grp_msk_group_crop)
            disk_group_crop = np.flipud(disk_group_crop)
            excentricity_group_crop = np.flipud(excentricity_group_crop)
            confidence_group_crop = np.flipud(confidence_group_crop)
            grp_confidence_group_crop = np.flipud(grp_confidence_group_crop)
        
        kwargs['image'] = img_group_crop.copy()
        kwargs['mask'] = msk_group_crop.copy()
        kwargs['group_mask'] = grp_msk_group_crop.copy()
        kwargs['solar_disk'] = disk_group_crop.copy()
        kwargs['excentricity_map'] = excentricity_group_crop.copy()
        kwargs['confidence_map'] = confidence_group_crop.copy()
        kwargs['group_confidence_map'] = grp_confidence_group_crop.copy()

        return kwargs

class DeepsunClassifRandomRotate(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def __call__(self, *args, force_apply=False, **kwargs):

        img_group_crop = kwargs['image'].copy()
        msk_group_crop = kwargs['mask'].copy()
        grp_msk_group_crop = kwargs['group_mask'].copy()
        disk_group_crop = kwargs['solar_disk'].copy()
        excentricity_group_crop = kwargs['excentricity_map'].copy()
        confidence_group_crop = kwargs['confidence_map'].copy()
        grp_confidence_group_crop = kwargs['group_confidence_map'].copy()

        # generate parameters to rotate (0, 90, 180, 270)
        angle = np.random.choice([0, 90, 180, 270])

        # rotate image
        if angle != 0:
            img_group_crop = np.rot90(img_group_crop, angle//90)
            msk_group_crop = np.rot90(msk_group_crop, angle//90)
            grp_msk_group_crop = np.rot90(grp_msk_group_crop, angle//90)
            disk_group_crop = np.rot90(disk_group_crop, angle//90)
            excentricity_group_crop = np.rot90(excentricity_group_crop, angle//90)
            confidence_group_crop = np.rot90(confidence_group_crop, angle//90)
            grp_confidence_group_crop = np.rot90(grp_confidence_group_crop, angle//90)

        kwargs['image'] = img_group_crop.copy()
        kwargs['mask'] = msk_group_crop.copy()
        kwargs['group_mask'] = grp_msk_group_crop.copy()
        kwargs['solar_disk'] = disk_group_crop.copy()
        kwargs['excentricity_map'] = excentricity_group_crop.copy()
        kwargs['confidence_map'] = confidence_group_crop.copy()
        kwargs['group_confidence_map'] = grp_confidence_group_crop.copy()
        
        return kwargs

 



class DeepsunRotateAndCropAroundGroup_Focus_Move(DualTransform):

    def __init__(self, standard_height, standard_width, 
                        focus_on_group=True,
                        random_move=False, random_move_percent=0.1,  
                        always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.standard_height = standard_height
        self.standard_width = standard_width

        self.focus_on_group = focus_on_group
        self.random_move = random_move
        self.random_move_percent = random_move_percent

        self.index = 0

    def get_transform_init_args_names(self):
        return ("standard_height", "standard_width", "focus_on_group", "random_move", "random_move_percent")

    def expand_small_spots(self, msk):
        out_msk = msk.copy()
        label_img = label(out_msk)
        regions = regionprops(label_img)
        
        for r in regions:
            if r.area == 1:
                coords = r.coords[0]
                # print(coords)
                out_msk[coords[0]-1:coords[0]+1,coords[1]-1:coords[1]+1] = msk[coords[0],coords[1]]
                
        return out_msk

    # get mask contaoining only components at given coordinates using regionprops
    def get_mask_from_coords(self, mask, coords):
        m = np.zeros_like(mask)
        m = m.astype(np.uint8)
        m2 = mask.copy()
        m2[m2>0] = 1
        l = label(m2)
        # print(np.unique(l)[1:])
        for val in np.unique(l)[1:]:
            # Get contours
            contours, hierarchies = cv2.findContours((l==val).astype(np.uint8) , cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Get convex hulls of contours to avoid missing sunspots with exotic shapes
            hull_list = []
            for i in range(len(contours)):
                hull = cv2.convexHull(contours[i])
                hull_list.append(hull)

            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(l==val.astype(np.uint8))
            # for c in contours:
            #     ax[0].plot(c[:,:,0], c[:,:,1], c='r')
            # plt.show()
            # print(contours)

            # print(coords)
            # Check if any of the coordinates is inside the convex hulls
            for c in coords:
                for cnt in hull_list:

                    # If the point is inside the convex hull, add corresponding sunspot to mask
                    if cv2.pointPolygonTest(cnt, (c[1], c[0]), False) >= 0:
                        m += (l==val).astype(np.uint8)
                        break


        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(mask)
        # ax[0].scatter([c[1] for c in coords], [c[0] for c in coords], c='r', s=10)
        # ax[1].imshow(m)
        # plt.show()

        return m

    def padder(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def get_bounding_box_around_group_with_padding(self, mask, offset):
        # Get the bounding box around non-zero pixels in mask
        x, y = np.nonzero(mask)
        # print(x, y)
        x1, x2 = (np.min(x), np.max(x)) if len(x) > 0 else (None, None)
        y1, y2 = (np.min(y), np.max(y)) if len(y) > 0 else (None, None)

        if (x1 is None) or (y1 is None):
            return 0, mask.shape[0]-1, 0, mask.shape[1]-1

        # Add padding
        x1 -= offset
        x2 += offset
        y1 -= offset
        y2 += offset


        # Make sure the bounding box is not outside the image
        x1 = max(x1, 0)
        x2 = min(x2, mask.shape[0])
        y1 = max(y1, 0)
        y2 = min(y2, mask.shape[1])

        return x1, x2, y1, y2

    def adapt_bbox_to_image_size(self, bbox, image_size):
        bbox_center = ((bbox[0] + bbox[1]) // 2, (bbox[2] + bbox[3]) // 2)
        bbox_size = (bbox[1] - bbox[0], bbox[3] - bbox[2])

        # if bbox is too small, expand it
        minimal_percentage = .4

        bbox_size = (max(bbox_size[0], image_size[0] * minimal_percentage),
                     max(bbox_size[1], image_size[1] * minimal_percentage))
        
        return (int(bbox_center[0] - bbox_size[0] // 2), int(bbox_center[0] + bbox_size[0] // 2),
                int(bbox_center[1] - bbox_size[1] // 2), int(bbox_center[1] + bbox_size[1] // 2))
    
    def data_aug_random_move(self, bbox, max_offset):
        '''
        Randomly move the bounding box
        param bbox: bounding box
        param max_offset: maximum offset in portion of the bbox size
        '''
        # Randomly move the bounding box
        x1, x2, y1, y2 = bbox
        horizontal_offset = (np.random.random(1) * 2*max_offset) - max_offset
        vertical_offset = (np.random.random(1) * 2*max_offset) - max_offset
        # print(f'horizontal_offset: {horizontal_offset}, vertical_offset: {vertical_offset}')

        x1 += int(horizontal_offset * (bbox[1] - bbox[0]))
        x2 += int(horizontal_offset * (bbox[1] - bbox[0]))
        y1 += int(vertical_offset * (bbox[3] - bbox[2]))
        y2 += int(vertical_offset * (bbox[3] - bbox[2]))
        
        return x1, x2, y1, y2

    def padding(self, array, xx, yy):
        """
        :param array: numpy array
        :param xx: desired height
        :param yy: desirex width
        :return: padded array
        """

        h = array.shape[0]
        w = array.shape[1]

        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

    def crop_img(self, img, bbox):
        # Crop image
        x1, x2, y1, y2 = bbox
        img = img[x1:x2, y1:y2]
        return img

    # crop image using bounding box and pad it to standard size
    def crop_and_pad(self, img, bbox, image_size):
        # Crop image
        img = self.crop_img(img, bbox)

        # Pad image
        img = self.padding(img, image_size[0], image_size[1])

        return img

    def print_elapsed_time(self, st, msg):
        end_time = time.time()
        print(f'DeepsunRotateAndCropAroundGroup Elapsed time {msg}: {end_time - st}')

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')

        # global_st = time.time()

        img = kwargs['image']
        msk = kwargs['mask']
        disk = kwargs['solar_disk'].astype(np.uint8)
        excentricity = kwargs['excentricity_map']
        confidence = kwargs['confidence_map']



        # print(disk.dtype)

        try:
            st = time.time()
            # Make sure that all the single-pixels spots in mask are expanded to 3x3
            msk = self.expand_small_spots(msk)
            # self.print_elapsed_time(st, 'expand_small_spots')


            # 1) Correct solar Angle ->  Rotate image + Zoom In
            angle = kwargs['solar_angle']
            deltashapeX = kwargs['deltashapeX']
            deltashapeY = kwargs['deltashapeY']
            # print('solar_angle', angle)

            # st = time.time()

            # # rot_img = rotate(img, angle=angle, reshape=True)
            # # rot_msk = rotate(msk, angle=angle, reshape=True)
            # #rot_img = rotate_CV_bound(img, angle=angle, interpolation=cv2.INTER_LINEAR)
            # #rot_msk = rotate_CV_bound(msk, angle=angle, interpolation=cv2.INTER_LINEAR)
            # rot_img = rotate_CV_bound(img, angle=angle, interpolation=cv2.INTER_NEAREST)
            # rot_msk = rotate_CV_bound(msk, angle=angle, interpolation=cv2.INTER_NEAREST)
            # rot_disk = rotate_CV_bound(disk, angle=angle, interpolation=cv2.INTER_NEAREST)
            # rot_excentricity = rotate_CV_bound(excentricity, angle=angle, interpolation=cv2.INTER_NEAREST)
            # rot_confidence = rotate_CV_bound(confidence, angle=angle, interpolation=cv2.INTER_NEAREST)
            # print('img.shape', img.shape)
            # print('msk.shape', msk.shape)
            # print('disk.shape', disk.shape)
            # print('excentricity.shape', excentricity.shape)
            # print('confidence.shape', confidence.shape)
            tmp_concat = np.concatenate((img[:,:,None], msk[:,:,None], 
                                        disk[:,:,None], excentricity[:,:,None], 
                                        confidence[:,:,None]), axis=-1)
            rot_concat = rotate_CV_bound(tmp_concat, angle=angle, interpolation=cv2.INTER_NEAREST)
            # print('rot_concat.shape', rot_concat.shape)
            rot_img = rot_concat[:, :, 0]
            rot_msk = rot_concat[:, :, 1]
            rot_disk = rot_concat[:, :, 2]
            rot_excentricity = rot_concat[:, :, 3]
            rot_confidence = rot_concat[:, :, 4]
            # self.print_elapsed_time(st, 'rotate')


            rot_img_zoom = rot_img[deltashapeX//2:rot_img.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_img.shape[1]-deltashapeY//2] 
            rot_msk_zoom = rot_msk[deltashapeX//2:rot_msk.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_msk.shape[1]-deltashapeY//2] 
            rot_disk_zoom = rot_disk[deltashapeX//2:rot_disk.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_disk.shape[1]-deltashapeY//2]
            rot_excentricity_zoom = rot_excentricity[deltashapeX//2:rot_excentricity.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_excentricity.shape[1]-deltashapeY//2]
            rot_confidence_zoom = rot_confidence[deltashapeX//2:rot_confidence.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_confidence.shape[1]-deltashapeY//2]

            # print(rot_img_zoom.shape, rot_msk_zoom.shape)
            assert rot_img_zoom.shape == rot_msk_zoom.shape

            
            # st = time.time()
            # grp_mask = self.get_mask_from_coords(rot_msk_zoom, kwargs['members'])
            grp_mask = self.get_mask_from_coords(rot_confidence_zoom, kwargs['members'])
            grp_rot_msk_zoom = rot_msk_zoom * grp_mask
            grp_rot_confidence_zoom = rot_confidence_zoom * grp_mask
            # self.print_elapsed_time(st, 'get_mask_from_coords')

            # 2) Crop around group
            # group_centroid = np.array(kwargs['centroid_px'])
            group_centroid = np.array((kwargs['members_mean_px'][1], kwargs['members_mean_px'][0]))

            # print(group_centroid)

            # minX = int(group_centroid[0])-self.standard_width//2
            # maxX = int(group_centroid[0])+self.standard_width//2
            # minY = int(group_centroid[1])-self.standard_height//2
            # maxY = int(group_centroid[1])+self.standard_height//2

            # img_group_crop = rot_img_zoom[minX:maxX,minY:maxY]
            # msk_group_crop = rot_msk_zoom[minX:maxX,minY:maxY]

            minX = self.standard_height + (int(group_centroid[1])-self.standard_width//2)
            maxX = self.standard_height + (int(group_centroid[1])+self.standard_width//2)
            minY = self.standard_height + (int(group_centroid[0])-self.standard_height//2)
            maxY = self.standard_height + (int(group_centroid[0])+self.standard_height//2)

            # st = time.time()
            
            # pad_rot_img_zoom = np.pad(rot_img_zoom, self.standard_height, self.padder, padder=0)
            # pad_rot_msk_zoom = np.pad(rot_msk_zoom, self.standard_height, self.padder, padder=0)
            # pad_grp_rot_msk_zoom = np.pad(grp_rot_msk_zoom, self.standard_height, self.padder, padder=0)
            # pad_rot_disk_zoom = np.pad(rot_disk_zoom, self.standard_height, self.padder, padder=0)
            # pad_rot_excentricity_zoom = np.pad(rot_excentricity_zoom, self.standard_height, self.padder, padder=0)
            # pad_rot_confidence_zoom = np.pad(rot_confidence_zoom, self.standard_height, self.padder, padder=0)
            # pad_grp_rot_confidence_zoom = np.pad(grp_rot_confidence_zoom, self.standard_height, self.padder, padder=0)

            pad_rot_img_zoom = np.pad(rot_img_zoom, self.standard_height, 'constant')
            pad_rot_msk_zoom = np.pad(rot_msk_zoom, self.standard_height, 'constant')
            pad_grp_rot_msk_zoom = np.pad(grp_rot_msk_zoom, self.standard_height, 'constant')
            pad_rot_disk_zoom = np.pad(rot_disk_zoom, self.standard_height, 'constant')
            pad_rot_excentricity_zoom = np.pad(rot_excentricity_zoom, self.standard_height, 'constant')
            pad_rot_confidence_zoom = np.pad(rot_confidence_zoom, self.standard_height, 'constant')
            pad_grp_rot_confidence_zoom = np.pad(grp_rot_confidence_zoom, self.standard_height, 'constant')

            # print('rot_img_zoom.shape', rot_img_zoom.shape)
            # tmp_concat = np.concatenate((rot_img_zoom[:,:,None], rot_msk_zoom[:,:,None], grp_rot_msk_zoom[:,:,None], 
            #                             rot_disk_zoom[:,:,None], rot_excentricity_zoom[:,:,None], rot_confidence_zoom[:,:,None], 
            #                             grp_rot_confidence_zoom[:,:,None]), axis=-1)
            # tmp_w = self.standard_height
            # tmp_concat = np.pad(tmp_concat, ((tmp_w,tmp_w),(tmp_w,tmp_w),(0,0)), 'constant')
            # pad_rot_img_zoom = tmp_concat[:,:,0]
            # pad_rot_msk_zoom = tmp_concat[:,:,1]
            # pad_grp_rot_msk_zoom = tmp_concat[:,:,2]
            # pad_rot_disk_zoom = tmp_concat[:,:,3]
            # pad_rot_excentricity_zoom = tmp_concat[:,:,4]
            # pad_rot_confidence_zoom = tmp_concat[:,:,5]
            # pad_grp_rot_confidence_zoom = tmp_concat[:,:,6]
            # self.print_elapsed_time(st, 'pad')


            img_group_crop = pad_rot_img_zoom[minX:maxX,minY:maxY]
            msk_group_crop = pad_rot_msk_zoom[minX:maxX,minY:maxY]
            grp_msk_group_crop = pad_grp_rot_msk_zoom[minX:maxX,minY:maxY]
            disk_group_crop = pad_rot_disk_zoom[minX:maxX,minY:maxY]
            excentricity_group_crop = pad_rot_excentricity_zoom[minX:maxX,minY:maxY]
            confidence_group_crop = pad_rot_confidence_zoom[minX:maxX,minY:maxY]
            grp_confidence_group_crop = pad_grp_rot_confidence_zoom[minX:maxX,minY:maxY]

            # st = time.time()
            # Get the bounding box around the group
            bbox = self.get_bounding_box_around_group_with_padding(grp_msk_group_crop, 10)
            # self.print_elapsed_time(st, 'get_bounding_box_around_group_with_padding')

            # Crop the image and mask around the group while keeping the same size
            if self.focus_on_group:
                # focus on the group
                # Modify the bounding box if data augmentation is enabled
                if self.random_move:
                    # print('random_move')
                    bbox = self.data_aug_random_move(bbox, max_offset=self.random_move_percent)
                                        
                    # Make sure the bounding box is not outside the image
                    x1, x2, y1, y2 = bbox
                    x1 = max(x1, 0)
                    x2 = min(x2, self.standard_width)
                    y1 = max(y1, 0)
                    y2 = min(y2, self.standard_height)
                    bbox = x1, x2, y1, y2
                    
                bbox = self.adapt_bbox_to_image_size( bbox, (self.standard_height, self.standard_width))
                img_group_crop = self.crop_and_pad(img_group_crop, bbox, (self.standard_height, self.standard_width))
                msk_group_crop = self.crop_and_pad(msk_group_crop, bbox, (self.standard_height, self.standard_width))
                grp_msk_group_crop = self.crop_and_pad(grp_msk_group_crop, bbox, (self.standard_height, self.standard_width))
                disk_group_crop = self.crop_and_pad(disk_group_crop, bbox, (self.standard_height, self.standard_width))
                excentricity_group_crop = self.crop_and_pad(excentricity_group_crop, bbox, (self.standard_height, self.standard_width))
                confidence_group_crop = self.crop_and_pad(confidence_group_crop, bbox, (self.standard_height, self.standard_width))
                grp_confidence_group_crop = self.crop_and_pad(grp_confidence_group_crop, bbox, (self.standard_height, self.standard_width))
                # print('focus on group')
            else:
                if self.random_move:
                    frac = np.max([(bbox[1]-bbox[0]) /self.standard_height, (bbox[3]-bbox[2]) /self.standard_width])
                    frac = np.sqrt(frac)
                    # print(minX, maxX, minY, maxY , frac, self.random_move_percent)
                    bbox = self.data_aug_random_move([minX,maxX,minY,maxY], max_offset=self.random_move_percent*frac)
                    minX,maxX,minY,maxY = bbox
                    # print(minX, maxX, minY, maxY )
                    
                    img_group_crop = pad_rot_img_zoom[minX:maxX,minY:maxY]
                    msk_group_crop = pad_rot_msk_zoom[minX:maxX,minY:maxY]
                    grp_msk_group_crop = pad_grp_rot_msk_zoom[minX:maxX,minY:maxY]
                    disk_group_crop = pad_rot_disk_zoom[minX:maxX,minY:maxY]
                    excentricity_group_crop = pad_rot_excentricity_zoom[minX:maxX,minY:maxY]
                    confidence_group_crop = pad_rot_confidence_zoom[minX:maxX,minY:maxY]
                    grp_confidence_group_crop = pad_grp_rot_confidence_zoom[minX:maxX,minY:maxY]

            # self.print_elapsed_time(st, 'focusMove Operations')
            

            assert img_group_crop.shape == msk_group_crop.shape

        except AssertionError:
            print("error")
            pass
            # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
            # img = kwargs['image'].copy()
            # ax[0].set_title(kwargs["name"])
            # ax[1].set_title(kwargs["centroid_px"])
            # ax[0].imshow(img, interpolation=None, cmap='gray')
            # ax[0].imshow(msk, interpolation=None, alpha=0.5)
            # ax[1].imshow(rot_img, interpolation=None, cmap='gray')
            # ax[1].imshow(rot_msk, interpolation=None, alpha=0.5)
            # ax[2].imshow(img_group_crop, interpolation=None, cmap='gray')
            # ax[2].imshow(msk_group_crop, interpolation=None, alpha=0.5)
            
            plt.show()
        
        # print(msk_group_crop.shape)
        # st = time.time()
        kwargs.pop('solar_angle',None)
        kwargs.pop('deltashapeX',None)
        kwargs.pop('deltashapeY',None)
        # kwargs.pop('centroid_px',None)
        # self.print_elapsed_time(st, 'pop operations')

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        # ax[0].imshow(img, interpolation=None, cmap='gray')
        # ax[0].imshow(msk, interpolation=None, alpha=0.5)
        # ax[1].imshow(rot_img_zoom, interpolation=None, cmap='gray')
        # ax[1].imshow(rot_msk_zoom, interpolation=None, alpha=0.5)
        # ax[2].imshow(img_group_crop, interpolation=None, cmap='gray')
        # ax[2].imshow(msk_group_crop, interpolation=None, alpha=0.5)
        # ax[1].scatter(group_centroid[0],group_centroid[1], c='r', s=1 )
        # plt.savefig(f'./test_classification_{self.index}.png', dpi=150)
        
        self.index+=1

        # st = time.time()
        kwargs['image'] = img_group_crop.copy()
        kwargs['mask'] = msk_group_crop.copy()
        kwargs['group_mask'] = grp_msk_group_crop.copy()
        kwargs['solar_disk'] = disk_group_crop.copy()
        kwargs['excentricity_map'] = excentricity_group_crop.copy()
        kwargs['confidence_map'] = confidence_group_crop.copy()
        kwargs['group_confidence_map'] = grp_confidence_group_crop.copy()
        # self.print_elapsed_time(st, 'last Copy Operations')

        # As groups in a single batch can have different number of members, 
        # we need to remove the 'members' key from the kwargs so that they can be stacked.
        kwargs.pop('members',None)


        # global_et = time.time()
        # print(f'DeepsunRotateAndCropAroundGroup time: {global_et-global_st} seconds')
        
        
        return kwargs


from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes

class DeepsunScaleWhitelight(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(self, p=p)
        self.index = 0
    
    def __call__(self, *args, force_apply=False, **kwargs):
        image = kwargs['image'].copy()
        disk = kwargs['solar_disk'].astype(np.uint8) // 255

        # scale part of image in solar disk to [0,1] range
        image[disk==1] = image[disk==1] / np.max(image[disk==1])
        # set rest of image to 0
        image[disk==0] = -1

        kwargs['image'] = image.copy()
        return kwargs

class DeepsunScaleExcentricityMap(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(self, p=p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):
        excentricity_map = kwargs['excentricity_map'].copy()
        disk = kwargs['solar_disk'].astype(np.uint8)// 255

        # scale part of image in solar disk to [0,1] range
        excentricity_map[disk==1] = excentricity_map[disk==1] / np.max(excentricity_map[disk==1])
        # set rest of image to 0
        excentricity_map[disk==0] = -1

        kwargs['excentricity_map'] = excentricity_map.copy()

        return kwargs

class DeepsunScaleConfidenceMap(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(self, p=p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):
        # DON'T USE this class
        # confidence_map is scaled by default, no need to do anything

        confidence_map = kwargs['confidence_map'].copy()
        disk = kwargs['solar_disk'].astype(np.uint8)// 255

        # scale part of image in solar disk to [0,1] range
        # confidence_map[disk==1] = confidence_map[disk==1] / np.max(confidence_map[disk==1])

        confidence_map[disk==1] = confidence_map[disk==1] / 1.0
        # set rest of image to 0
        confidence_map[disk==0] = 0

        kwargs['confidence_map'] = confidence_map.copy()

        return kwargs




class DeepsunImageAndScalars2ThreeChannels(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(self, p=p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):
        img = kwargs['image'] 
        img = img / np.max(img) # to [0,1] range

        angular_excentricity = kwargs['angular_excentricity'] / 90 # to [0,1] range

        centroid_Lat = (kwargs['centroid_Lat'] * (180/np.pi)) # to degrees -> [-90,90]
        centroid_Lat = (centroid_Lat + 90) / 180 # to [0,1] range


        excent_img = np.ones_like(img)*angular_excentricity
        Lat_img = np.ones_like(img)*centroid_Lat

        # print(f'img: {img.shape},  excent_img: {excent_img.shape}, Lat_img: {Lat_img.shape}')

        img_3channels = np.stack((img, excent_img, Lat_img), axis=0)
        # img_3channels = np.stack((excent_img, img,  Lat_img), axis=0)
        # img_3channels = np.stack((excent_img, Lat_img, img ), axis=0)

        # print(f'img_3channels= {img_3channels.shape}')

        kwargs["image"] = img_3channels

        # fig = plt.figure()
        # ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8], pad=0.0)
        # r, g, b = img_3channels[0,:,:], img_3channels[1,:,:], img_3channels[2,:,:]
        # ax.imshow_rgb(r, g, b)
        # plt.savefig(f'./test_3Channels_{self.index}.png', dpi=150)

        self.index+=1


        return kwargs

class DeepsunImageMaskProduct(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')
        img = kwargs['image'].copy()
        msk = kwargs['mask'].copy()

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        # ax[0].imshow(img, interpolation=None, cmap='gray')
        # ax[1].imshow(msk, interpolation=None)
        # ax[2].imshow(img * msk, interpolation=None, cmap='gray')
        # plt.savefig(f'./test_classification_MaskIm{self.index}.png', dpi=150)

        self.index+=1

        kwargs['image'] = img * msk

        return kwargs

class DeepsunMcIntoshScaleAdditionalInfo(BasicTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):


        angular_excentricity = kwargs['angular_excentricity'] / 90 # to [0,1] range

        centroid_Lat = (kwargs['centroid_Lat'] * (180/np.pi)) # to degrees -> [-90,90]
        centroid_Lat = (centroid_Lat + 90) / 180 # to [0,1] range

        kwargs['angular_excentricity'] = angular_excentricity
        kwargs['centroid_Lat'] = centroid_Lat

        return kwargs


class DeepsunMcIntoshImageSingle2ThreeChannels(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(self, p=p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):
        img = kwargs['image'] 
        img = img / np.max(img) # to [0,1] range

        img_3channels = np.stack((img,)*3, axis=0)
        kwargs["image"] = img_3channels

        self.index+=1

        return kwargs

class DeepsunMaskLargestSpotOLD(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.index = 0

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')
        msk = kwargs['mask'].copy()
        # print(np.unique(msk))
        # print([k for k in kwargs])
        

        try:

            label_img = label(msk)
            regions = regionprops(label_img)
            assert(len(regions)>0)
            sizes = [r.area for r in regions]
            # print(len(sizes))
            largest = np.argmax(sizes)
            l = regions[largest].label
            msk[label_img != l] = 0

        except ValueError:
            print('Erreur')
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
            img = kwargs['image'].copy()
            ax[0].set_title(kwargs["name"])
            ax[1].set_title(kwargs["centroid_px"])
            ax[0].imshow(img[0,:,:], interpolation=None, cmap='gray')
            ax[1].imshow(kwargs['mask'], interpolation=None)
            ax[2].imshow(msk, interpolation=None, cmap='gray')
            plt.savefig(f'./test_largestSpot{self.index}.png', dpi=150)

            
            raise

        # label_img = label(msk)
        # regions = regionprops(label_img)
        # sizes = [r.area for r in regions]
        # largest = np.argmax(sizes)
        # l = regions[largest].label
        # msk[label_img != l] = 0

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        # img = kwargs['image'].copy()
        # ax[0].imshow(img[0,:,:], interpolation=None, cmap='gray')
        # ax[1].imshow(kwargs['mask'], interpolation=None)
        # ax[2].imshow(msk, interpolation=None, cmap='gray')
        # plt.savefig(f'./test_largestSpot{self.index}.png', dpi=150)

        kwargs['mask'] =  msk

        self.index +=1

        return kwargs

class DeepsunMaskLargestSpot(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.index = 0

    def largest_in_mask(self, msk):
        try:
            msk[msk>0] = 1
            label_img = label(msk)
            regions = regionprops(label_img)
            assert(len(regions)>0)
            sizes = [r.area for r in regions]
            # print(len(sizes))
            largest = np.argmax(sizes)
            l = regions[largest].label
            msk[label_img != l] = 0

        except ValueError:
            print('Erreur')
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
            img = kwargs['image'].copy()
            ax[0].set_title(kwargs["name"])
            ax[1].set_title(kwargs["centroid_px"])
            ax[0].imshow(img[0,:,:], interpolation=None, cmap='gray')
            ax[1].imshow(kwargs['mask'], interpolation=None)
            ax[2].imshow(msk, interpolation=None, cmap='gray')
            plt.savefig(f'./test_largestSpot{self.index}.png', dpi=150)

            raise
        except AssertionError:
            # print('Empty mask')
            msk = np.zeros_like(msk)

            
        return msk

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')

        msk1 = kwargs['mask'].copy()
        # print(np.unique(msk))
        # print([k for k in kwargs])
        kwargs['mask_LargestSpot'] =  self.largest_in_mask(msk1)

        if 'group_mask' in kwargs:
            msk2 = kwargs['group_mask'].copy()
            kwargs['group_mask_LargestSpot'] =  self.largest_in_mask(msk2)

        self.index +=1

        return kwargs


class DeepsunMaskToOneHot(DualTransform):
    def __init__(self, classes, always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.index = 0
        self.classes = classes

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')
        msk = kwargs['mask'].copy()
        msk[msk>0]=1
        

        # try:

        #     label_img = label(msk)
        #     regions = regionprops(label_img)
        #     # assert(len(regions)>0)
        #     sizes = [r.area for r in regions]
        #     # print(len(sizes))
        #     largest = np.argmax(sizes)
        #     l = regions[largest].label
        #     msk[label_img != l] = 0

        # except AssertionError:
        #     print('Erreur')
        #     # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        #     # img = kwargs['image'].copy()
        #     # ax[0].set_title(kwargs["name"])
        #     # ax[1].set_title(kwargs["centroid_px"])
        #     # ax[0].imshow(img, interpolation=None, cmap='gray')
        #     # ax[1].imshow(kwargs['mask'], interpolation=None)
        #     # ax[2].imshow(msk, interpolation=None, cmap='gray')
        #     # plt.savefig(f'./test_largestSpotPenUm{self.index}.png', dpi=150)

            
        #     raise
        
        # One-hot version of the mask
        # msk_one_hot = (np.arange(kwargs['mask'].max()+1) == kwargs['mask'][...,None]).transpose(2, 0, 1).astype(int)
        msk_one_hot = (np.arange(len(self.classes)+1) == kwargs['mask'][...,None]).transpose(2, 0, 1).astype(int)
        
        # #we care only about the largest spot
        kwargs['mask_one_hot'] = msk_one_hot #* msk

        self.index +=1

        return kwargs

class DeepsunCLAHE(DualTransform):
    def __init__(self, always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.index = 0
    
    def __call__(self, *args, force_apply=False, **kwargs):
        img = kwargs['image'].copy().astype(np.uint16)

        # print(img.dtype)
        clahe = cv2.createCLAHE(clipLimit=100.0,tileGridSize=(8,8))

        clahe_img = clahe.apply(img)
        # print(clahe_img.dtype)
        kwargs['CLAHE'] = clahe_img.astype(float)
        self.index+=1

        return kwargs
