#!/usr/bin/env python
# coding: utf-8

from .create_detection_masks import *
from skimage import data, filters, segmentation

from skimage import data
import skimage.morphology as morphology
from skimage.morphology import disk, ball
from skimage.morphology import square, disk
import numpy as np
from skimage.measure import label, regionprops

from scipy.signal import find_peaks
from scipy import interpolate
from scipy import ndimage

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image, ImageFilter

def search_max_threshold(region, tmp2):
    thresh_history = []
    history_pen = []
    history_um = []
    
#     farthest_peak : ? 
    hist_val, hist_edges = np.histogram(tmp2, bins=255)  

    peaks, properties = find_peaks(hist_val,prominence=200)
    
#     print(peaks)
    
    if len(peaks) == 0:
        return [0,0]
    if len(np.unique(tmp2)) < 3: 
        return [0 , 0]
    
    min_th = 0
    init_max_th = ( hist_edges[peaks[-1]] + np.min(tmp2) ) /2
    
    ok = False
    max_th = init_max_th

    while (max_th < hist_edges[peaks[-1]]) :     
        try:        
            tmp3 = tmp2[np.where((tmp2 >= min_th) & (tmp2 <= max_th))]
            
            if len(np.unique(tmp3)) < 3 :
                max_th += 150
                continue

            thresholds = filters.threshold_multiotsu(tmp3, classes = 3)     
    
            max_th += 150

            levels = np.digitize(tmp2, bins=thresholds)


            tmp_penumbrae = segmentation.clear_border(levels == 1)
            tmp_umbrae = segmentation.clear_border(levels == 0)

            label_pen = label(tmp_penumbrae)
            label_um = label(tmp_umbrae)

            pen_props = regionprops(label_pen)
            um_props = regionprops(label_um)

            if len(history_um) == 0 and len(history_pen) == 0:    
                history_pen.append(len(pen_props))
                history_um.append(len(um_props))
                thresh_history.append(thresholds)
                continue

            if (((len(history_um) > 0) and (len(um_props) - history_um[0] < 4  )) and
                ((len(history_pen) > 0) and (len(pen_props) - history_pen[-1] < 10  ))):
            
                history_pen.append(len(pen_props))
                history_um.append(len(um_props))
                thresh_history.append(thresholds) 
            else:
                return thresh_history[-1]

        except ValueError:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
            ax.hist(tmp2.ravel(), bins=255)
            fig.show()
            print(tmp3.shape)
            print(np.unique(tmp3))
            return 0,0
            # raise Error
            
    try:
        
        return thresh_history[-1]
    
    except IndexError:
        # print(f'min_th:{min_th}; init_max_th:{init_max_th}; max_th:{max_th}; tmp2.min:{np.min(tmp2)}')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        # ax.hist(tmp2.ravel(), bins=255)
        ax.imshow(region, cmap='gray')
        fig.show()
        return 0,0
        # raise Error

def local_hist_bbox(image, radius, bbox, padding):
    # Bounding box (min_row, min_col, max_row, max_col)
    init_region = image[bbox[0]-padding : bbox[2]+padding , bbox[1]-padding : bbox[3]+padding]
    img_cpy= image.copy()
    
    #original image
    region = img_cpy[bbox[0]-padding : bbox[2]+padding , bbox[1]-padding : bbox[3]+padding]
    
    
    center = [image.shape[0]//2, image.shape[1]//2]
    outside = create_circular_mask( image.shape[1], image.shape[0] ,center, radius)

    tmp = outside[bbox[0]-padding : bbox[2]+padding , bbox[1]-padding : bbox[3]+padding]

    
    tmp2 = np.stack((tmp[:,:,None], region[:,:, None]), axis =-1)
    tmp2 = np.max(tmp2, axis=-1).squeeze()

    thresholds = search_max_threshold(region, tmp2)
     
    levels = np.digitize(tmp2, bins=thresholds)
    
    levels = 2- levels
    
    return levels, thresholds

def generate_grid(image, n_cells_per_side = 6 ):

    cell_side = image.shape[0] // n_cells_per_side
    half_cell_side = cell_side //2
    list_x = [k * half_cell_side for k in range(1, 2*n_cells_per_side)]
    centers = np.repeat(np.array(list_x)[:,None], 2*n_cells_per_side -1 , axis = -1)

    bboxes = np.zeros((centers.shape[0],centers.shape[1],4))
    # print(bboxes.shape)
    centers= np.stack((centers, centers.T),axis=-1)
    for i in range(centers.shape[0]):
        for j in range(centers.shape[1]):
            center = centers[i,j]
            bboxes[i,j,:] = np.array([center[1]-half_cell_side, center[1]+half_cell_side, center[0]-half_cell_side, center[0]+half_cell_side])
    return bboxes, centers


def compute_cells_thresholds(bboxes, image, radius):

    thresholds = np.zeros_like(bboxes[:,:,:2])

    for i in range(thresholds.shape[0]):
        for j in range(thresholds.shape[1]):
            bbox = bboxes[i,j].astype(int)
            # print(bbox)
            levels, thresh = local_hist_bbox(image, radius, bbox, 0)
            thresholds[i,j] = thresh

    return thresholds


final_mask = None

def nearest_nonzero_idx_v2(a,x,y):
    tmp = a[x,y]
    a[x,y] = 0
    r,c = np.nonzero(a)
    
    if len(r) == 0 :
        return 0,0
    
    a[x,y] = tmp
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]


def replace_thresh_mean_valid(thresholds, thresh_pen, thresh_um):
    cpy = thresholds.copy()
   
    pen = cpy[:,:,1]
#     valid_penum = pen[pen != thresh_pen]
    mean_penum_thresh = thresholds[:,:,1].mean()

    pen[pen == int(thresh_pen)] = mean_penum_thresh
    
    um = cpy[:,:,0]
#     valid_um = um[um != thresh_um]
    mean_um_thresh = thresholds[:,:,0].mean()  
    um[um == int(thresh_um)] = mean_um_thresh
 
    return cpy
    

def replace_thresh_closest_valid(thresholds, thresh_pen, thresh_um):
    cpy = thresholds.copy()

    pen = cpy[:,:,1]
    pen[pen == int(thresh_pen)] = 0
    pen_cpy = pen.copy()
    
    for i in range(pen.shape[0]):
        for j in range(pen.shape[1]):
            if pen[i,j] == 0:
#                 print(i,j)
                i2,j2 = nearest_nonzero_idx_v2(pen_cpy, i,j)
                pen[i,j] = pen[i2,j2] 
    
    um = cpy[:,:,0]
    um[um == int(thresh_um)] = 0
    um_cpy = um.copy()
    
    for i in range(um.shape[0]):
        for j in range(um.shape[1]):
            if um[i,j] == 0:
                i2,j2 = nearest_nonzero_idx_v2(um_cpy, i,j)
                um[i,j] = um[i2,j2] 
                    
    return cpy


def replace_thresh2(thresholds, thresh_pen, thresh_um, max_offset):
    thresh_diff = thresh_pen - thresh_um
#     max_offset = 100
    # thresholds[i,j] = umbrae, penumbrae
    cpy = thresholds.copy()
    cpy_pen = cpy[:,:,1]
    cpy_pen[abs(cpy_pen - thresh_pen) > max_offset] = thresh_pen

    cpy_um = cpy[:,:,0]
    cpy_um[abs(cpy_um - thresh_um) > max_offset] = thresh_um

    return cpy

def display_thresholds2(image, centers, thresholds, size, opening_width):
    x_ = np.array([0] + np.squeeze(centers[:,1,0]).tolist() + [image.shape[0]-1])[:,None]

    tmp = np.array(np.meshgrid(x_,x_))
    tmp_2col = np.moveaxis(tmp,0,-1).reshape(-1, 2)

    tmp_thresh_lo = np.ones_like(tmp[0])*thresholds[:,:,0].mean()
    
    tmp_thresh_lo[1:-1,1:-1] = thresholds[:,:,0]

    x2 = np.linspace(0, image.shape[0], image.shape[0])
    y2 = np.linspace(0, image.shape[0], image.shape[0])

    r = np.linspace(0, image.shape[0], tmp_thresh_lo.shape[0] , endpoint=True)

    f = interpolate.interp2d(r, r, tmp_thresh_lo, kind='linear')
    Z2 = f(x2,y2)

    gaussian_lo = ndimage.filters.uniform_filter(Z2, size=size, mode='reflect', cval=1)

    tmp_thresh_hi = np.ones_like(tmp[0])*thresholds[:,:,1].mean()
    tmp_thresh_hi[1:-1,1:-1] = thresholds[:,:,1]

    f = interpolate.interp2d(r, r, tmp_thresh_hi, kind='linear')
    Z3 = f(x2,y2)

    gaussian_hi = ndimage.filters.uniform_filter(Z3, size=size, mode='reflect', cval=1)

    um_mask = image < gaussian_lo

    penumbrae_1= (image > gaussian_lo) 
    penumbrae_2= (image < gaussian_hi) 
    penum_mask = np.logical_and(penumbrae_1,penumbrae_2)
    
    out = penum_mask + um_mask*2

    out = clean_circle(out, opening_width)

    return out.astype(np.int32)

    
def f(pen_um_thresholds, size, max_offset, opening_width, centers, thresholds, image):
#     print(pen_um_thresholds)
    generic_thresh_pen = max(pen_um_thresholds)
    generic_thresh_um = min(pen_um_thresholds)

    new_thresh = replace_thresh2(thresholds, generic_thresh_pen, generic_thresh_um, max_offset)
#     new_thresh = replace_thresh_mean_valid(new_thresh, generic_thresh_pen, generic_thresh_um)
    new_thresh = replace_thresh_closest_valid(new_thresh, generic_thresh_pen, generic_thresh_um)
#     print(new_thresh)
    return display_thresholds2(image, centers,  new_thresh, size, opening_width)

def clean_circle(mask, opening_width):
    
    mask[0,:] = 0
    mask[mask.shape[0]-1,:] = 0
    mask[:,0] = 0
    mask[:,mask.shape[1]-1] = 0
    
    #first remove the big chunks of croissants
    borders = segmentation.clear_border(mask > 0)
    
    label_borders = label(borders)
    borders_props = regionprops(label_borders)
    for prop in borders_props:
        if prop.major_axis_length > mask.shape[0]//3:
            bbox = prop.bbox
            mask[bbox[0]:bbox[2],bbox[1]:bbox[3]][prop.image > 0 ] = 0
    
    #remove the small grains
    mask = skimage.morphology.opening(mask, disk(opening_width))


    return mask
