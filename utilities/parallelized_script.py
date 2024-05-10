#!/usr/bin/env python
# coding: utf-8

from ast import parse
import glob
import cv2
from skimage import data, filters, segmentation
import itertools

import time
from datetime import timedelta


from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
from skimage.measure import shannon_entropy, label, regionprops
from skimage.morphology import square, disk
from skimage.io import imread

from scipy.signal import find_peaks
import numpy as np

import os, sys
# print(os.path.abspath('..'))
# sys.path.append(os.path.abspath('..'))
from create_detection_masks import *
import Otsu_Grid_utilities as otsu_grid
import Otsu_Local_utilities as otsu_local

import concurrent.futures

import tqdm

def dump_mask(dest_filename, mask):
    m2 = mask.astype(np.uint8)
    # print(m2.dtype, np.unique(m2))
    # print(dest_filename)
    out_im = Image.fromarray(m2, mode='L')
    out_im.save(dest_filename)
    
def load_file(filename):
    basename = os.path.basename(filename)
    base_no_ext = basename.split('.')[0]
    # print(basename)

    header = get_FITS_header(filename)
    # wl_resized, padding = dump_FITS_image(filename, header['SOLAR_P0'])
    wl_resized  = imread(filename)

    # center = [header['CENTER_X'],header['CENTER_Y']]
    center = [wl_resized.shape[0]//2,wl_resized.shape[1]//2]
    # print(center)
    radius = header['SOLAR_R']

    pixel_nb = header['NAXIS1']
    pix_bit = pow(2, 12) - 1 
    poly_order = 2

    # centered images
    xce = int(pixel_nb/2)
    yce = int(pixel_nb/2)

    pixMat_flat = clv_correction(wl_resized,
                                header['SOLAR_R'],
                                xce,
                                yce,
                                poly_order,
                                pix_bit)

    return pixMat_flat, center, radius


def apply_local(image, radius, thresholds=[2500, 3100], padding=35):
    
    local_mask = otsu_local.bbox_f(np.array(thresholds), padding, image, radius)

    return local_mask
    # horiz_flip =  np.flip(local_mask, axis = 0)
    # return horiz_flip

def apply_local_tophat(image, radius, tophat_thresh=500, tophat_radius=35, padding=35):

    # print('ok')
    
    try:
        local_mask = otsu_local.bbox_f_tophat(tophat_thresh, tophat_radius, padding, image, radius)
    except ValueError:
        raise ValueError

    return local_mask
    # # horiz_flip =  np.flip(local_mask, axis = 0)
    # return horiz_flip


def apply_grid(image, radius, cells_per_side, thresholds=[2500, 3100], offset=150, smoothing=20, opening=2):
    
    bboxes, centers = otsu_grid.generate_grid(image, cells_per_side)
    threshold_grid = otsu_grid.compute_cells_thresholds(bboxes, image, radius)
    
    grid_mask = otsu_grid.f(np.array(thresholds), smoothing, offset, opening,
                            centers, threshold_grid, image)
    
    return grid_mask
    # horiz_flip =  np.flip(grid_mask, axis = 0)
    # return horiz_flip

def find_upper_thresh(image):
    init = 100000
    cur = init
    scaling = 0.8
    last_peak = 0
    while last_peak < 1000:
        hist_val, hist_edges = np.histogram(image, bins=255)
        peaks, properties = find_peaks(hist_val,prominence=cur)  

        cur = cur*scaling
        if len(peaks) == 0:
            continue
        last_peak = hist_edges[peaks[-1]]
    

    max_thresh = None
    for i in range(peaks[-1], 0, -1):
        if hist_val[i] < 0.05*hist_val[peaks[-1]]:
            max_thresh = hist_edges[i]
            break
            
    return max_thresh, last_peak

def apply_grid2(image, radius, cells_per_side,  thresholds_width= 500, offset=150, smoothing=20, opening=2):
    
    thresh, _ = find_upper_thresh(image)
    thresholds = [thresh - thresholds_width, thresh]
    bboxes, centers = otsu_grid.generate_grid(image, cells_per_side)
    threshold_grid = otsu_grid.compute_cells_thresholds(bboxes, image, radius)
    
    grid_mask = otsu_grid.f(np.array(thresholds), smoothing, offset, opening,
                            centers, threshold_grid, image)
    
    return grid_mask
    # horiz_flip =  np.flip(grid_mask, axis = 0)
    # return horiz_flip


def apply(filename, args):
    basename = os.path.basename(filename)
    base_no_ext = basename.split('.')[0]

    pixMat_flat, center, radius = load_file(filename)
    
    out_mask = None
    
    if args.method == 'local':    
        assert args.thresholds is not None
        assert args.padding is not None
        out_mask = apply_local(pixMat_flat, radius, args.thresholds, args.padding)
    if args.method == 'local_tophat':    
        assert args.tophat_thresh is not None
        assert args.tophat_radius is not None
        assert args.padding is not None
        try:
            out_mask = apply_local_tophat(pixMat_flat, radius, args.tophat_thresh, args.tophat_radius, args.padding)
        except ValueError:
            print(filename)

    elif args.method == 'grid':
        assert args.thresholds is not None
        assert args.offset is not None
        assert args.smoothing is not None
        assert args.cells_per_side is not None
        # print(pixMat_flat.min(), ' ',  pixMat_flat.max())
        out_mask = apply_grid(pixMat_flat, radius, args.cells_per_side, 
                              args.thresholds, args.offset, args.smoothing)

    elif args.method == 'grid2':
        assert args.offset is not None
        assert args.smoothing is not None
        assert args.cells_per_side is not None
        out_mask = apply_grid2(pixMat_flat, radius, args.cells_per_side, 
                              args.thresh_dist, args.offset, args.smoothing)
    
    local_filename = os.path.join(args.dest_dir , base_no_ext+ '.png')
    
    dump_mask(local_filename, out_mask)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate masks from Global/Grid/Local methods")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_images_dir', dest='images_dir', type=str, default=None, action='store', 
                            help='Directory where the original natural images are stored')
    input_group.add_argument('--input_images_list', dest='images_list', default=[], 
                            help='Directory where the original natural images are stored')
    
    parser.add_argument('--dest_dir', dest='dest_dir', required=True, type=str, 
                            help='Directory where the output masks will be stored')
    parser.add_argument('--method' , dest='method', required=True, type=str,
                            help='Method to apply to generate masks')

    parser.add_argument('--thresholds', dest = 'thresholds', required=False, nargs=2, type=int,
                            help='Intensity thresholds to use as reference.')
    
    # Arguments for Local Method
    parser.add_argument('--padding', dest='padding', required=False, default=None, type=int)
    
    # Arguments for Local Method using tophat transform
    parser.add_argument('--tophat_thresh', dest='tophat_thresh', required=False, default=None, type=int)
    parser.add_argument('--tophat_radius', dest='tophat_radius', required=False, default=None, type=int)


    # Arguments for Grid Method
    parser.add_argument('--cells_per_side', dest='cells_per_side', required=False, default=None, type=int)
    parser.add_argument('--offset', dest='offset', required=False, default=None, type=int)
    parser.add_argument('--smoothing', dest='smoothing', required=False, default=None, type=int)
    parser.add_argument('--opening', dest='opening', required=False, default=None, type=int)


    # Arguments for Grid2 Method
    parser.add_argument('--thresh_dist', dest='thresh_dist', required=False, default=None, type=int)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    images = []

    if args.images_dir is not None:
        images_dir = args.images_dir
        images = sorted(glob.glob(os.path.join(images_dir, '*.FTS')))

    else: 
        # user provided images_list argument
        pass

    # local_dest = '/home/sayez/DATASETS/automatic/local'
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    to_process = images
    # to_process = [(img, args) for img in images[:10]]
    # to_process = [(img, args) for img in images]

    executor = concurrent.futures.ProcessPoolExecutor()
    
    start = time.time()


    for i in tqdm.tqdm(executor.map(apply, to_process, itertools.repeat(args))):
        pass
    end = time.time()
    print(f"Total elapsed time {str(timedelta(seconds=end - start))}")

if __name__ == "__main__":
    main()

