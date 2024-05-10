#!/usr/bin/env python
# coding: utf-8

from .create_detection_masks import *
import glob
import cv2
from skimage import data, filters, segmentation

from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball, black_tophat, white_tophat
from skimage.measure import shannon_entropy, label, regionprops
from skimage.morphology import square, disk

from scipy.signal import find_peaks
import numpy as np

import matplotlib.patches as patches

# def search_max_threshold(region, tmp2, default_thresh):
    # thresh_history = [np.array([default_thresh.min(), default_thresh.max()])]
def search_max_threshold(region, tmp2):
    thresh_history = []
    history_pen = []
    history_um = []
    
#     farthest_peak : ? 
    hist_val, hist_edges = np.histogram(tmp2, bins=255)  

    min_th = 0
    
    peaks, properties = find_peaks(hist_val,prominence=200)
    
    if len(peaks) == 0:
        return [0,0]
    if len(np.unique(tmp2)) < 3: 
        return [0 , 0]
    
    ok = False    
#     init_max_th = hist_edges[len(hist_edges)//2]
    init_max_th = ( hist_edges[peaks[-1]] + np.min(tmp2) ) /2
    
    max_th = init_max_th
#     print(f'min_th:{min_th}; init_max_th:{init_max_th}; max_th:{max_th}; tmp2.min:{np.min(tmp2)}')
    while (max_th < hist_edges[peaks[-1]]) :
        try:
            tmp3 = tmp2[np.where((tmp2 >= min_th) & (tmp2 <= max_th))]
            
            if len(np.unique(tmp3)) < 3 :
                # if we grow beyond peak in this section and never went further
                # i.e. histories are still empty and we leave the loop
                # make sure that thresholds (so no foreground is found) are still returned.
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

        # except Error:
            # pass

        except ValueError:
            import collections

            print(f'min_th:{min_th}; peak: {hist_edges[peaks[-1]]}; init_max_th:{init_max_th}; max_th:{max_th}; tmp2.min:{np.min(tmp2)}')
            print(np.unique(tmp3))
            print(collections.Counter(tmp3))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
            ax.hist(tmp3.ravel(), bins=int(max(np.unique(tmp3))))
            plt.savefig('viewerror.png')
            # print("tmp3", tmp3.shape)
            print("search_max_threshold")
            print("tmp2", np.unique(tmp2))
            print("tmp3", np.unique(tmp3))
            return [0,0]
            # raise ValueError

    try:
        # check that at least one of the max_th was useful...if not, then we should return thresholds that do not
        # lead to foreground classes
        if len(thresh_history) == 0:
            return [0 , 0]

        return thresh_history[-1]
    
    except IndexError:
        print(f'min_th:{min_th}; peak: {hist_edges[peaks[-1]]}; init_max_th:{init_max_th}; max_th:{max_th}; tmp2.min:{np.min(tmp2)}')
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
        # # ax.hist(tmp2.ravel(), bins=255)
        # ax.imshow(region, cmap='gray')
        # fig.show()
        # return 0,0
        raise IndexError


def find_centroids2(image):
    contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    centers = []
    for contour in contours:   
        center = None
        
        if contour.shape[0] < 5 :
            continue
        else:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center = np.array([round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])])
            else:
                continue
                
        centers.append(center)
        
    return centers


def clean_circle(mask, opening_min_area, radius):
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
    
#     mask = skimage.morphology.area_opening(mask, area_threshold = opening_min_area)
    # mask = skimage.morphology.opening(mask, disk(3))
    mask = skimage.morphology.opening(mask, disk(opening_min_area))

    center = [mask.shape[0]//2, mask.shape[1]//2]
    outside_remover = create_circular_mask( mask.shape[1], mask.shape[0] ,center, radius)
    mask = mask*outside_remover     
    
    return mask


def local_hist_bbox(image, radius, bbox, padding, default_thresh=(2500,3100)):
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

    thresholds = default_thresh
    try:
        # thresholds = search_max_threshold(region, tmp2, default_thresh)
        thresholds = search_max_threshold(region, tmp2)
    except ValueError:
        raise ValueError
    except IndexError:
        print(bbox)
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='r', facecolor='none')

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 2))
        # ax.hist(tmp2.ravel(), bins=255)
        ax[0].imshow(image, cmap='gray')
        # Add the patch to the Axes
        ax[0].add_patch(rect)
        ax[1].imshow(region, cmap='gray')
        fig.show()
        raise Error

    levels = np.digitize(tmp2, bins=thresholds)
    
    levels = 2- levels
    
    return levels, thresholds

def find_bboxes(labels):
    props_labels = regionprops(labels)
    
    bboxes = []
    for prop in props_labels:
        if prop.area > 15:
            bbox = np.array(prop.bbox)
            bboxes.append(bbox)
#             print(f'area: {prop.area} ; bbox : {bbox}')

    return bboxes  

def bbox_f(pen_um_thresholds, padding, image, radius):
    padding = int(padding)
    
    im_penumbrae = segmentation.clear_border(image < pen_um_thresholds.max())
    im_umbrae = segmentation.clear_border(image < pen_um_thresholds.min())
    
    label_im_penumbrae = label(im_penumbrae)
    penumbrae_props_bboxes = find_bboxes(label_im_penumbrae)
    
    label_im_umbrae = label(im_umbrae)
    umbrae_props_bboxes = find_bboxes(label_im_umbrae)

    out_mask = np.zeros_like(image)
    
    # for bbox in tqdm.tqdm(penumbrae_props_bboxes):
    for bbox in penumbrae_props_bboxes:
        levels, _ = local_hist_bbox(image, radius, bbox, padding, np.array(pen_um_thresholds))
        out_mask[bbox[0]-padding : bbox[2]+padding , bbox[1]-padding : bbox[3]+padding] = levels
     
    out_mask = clean_circle(out_mask, 3, radius)

    return out_mask.astype(np.int32)

def bbox_f_tophat(tophat_threshold, tophat_radius, padding, image, radius):
    padding = int(padding)
#     print(padding)
    tophat_disk = disk(tophat_radius)

    spots = black_tophat(image, tophat_disk)
    spots_cpy = spots.copy()
    spots_cpy[spots_cpy<=tophat_threshold] = 0
    spots_cpy[spots_cpy>tophat_threshold] = 1
    
    label_im = label(spots_cpy)
    props_bboxes = find_bboxes(label_im)
    
    out_mask = np.zeros_like(image)
    
    for bbox in props_bboxes:
        try:
            levels, _ = local_hist_bbox(image, radius, bbox, padding, )

        except ValueError:
            raise ValueError
            
        out_mask[bbox[0]-padding : bbox[2]+padding , bbox[1]-padding : bbox[3]+padding] = levels
    
        
    out_mask = clean_circle(out_mask, 3, radius)
    
    return out_mask.astype(np.int32)



