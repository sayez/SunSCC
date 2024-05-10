import numpy as np
from numpy.polynomial import polynomial as P
import sunpy as sp
import sunpy.map as sunmap
import glob

from astropy.io import fits
import astropy.units as u

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
import math
import cv2
from PIL import Image

import skimage
from skimage import feature
from skimage import data, filters

from scipy import ndimage 
import sqlalchemy

import matplotlib.animation as animation
from matplotlib import rc
import matplotlib.patches as patches 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage.interpolation import rotate

import SimpleITK as sitk

import argparse

from skimage.segmentation import watershed
from scipy import ndimage as ndi


def cart2pol(x, y, center):
    r = np.sqrt(pow(x - center[0],2) + pow(y - center[1],2))
    theta = np.arctan2(y - center[1], x - center[0]) * 180/3.14
    return r, theta

def pol2cart(img, center):
    X, Y = np.meshgrid(np.arange(0, 2048), 
                       np.arange(0, 2048))
    R, theta = cart2pol(X, Y, center)
    R = R.astype(int)
    theta = theta.astype(int)    
    cart_img = img[theta, R]
    return cart_img

def polar2cart(r, theta, center):
    x = r  * np.cos(theta) + center[0]
    y = r  * np.sin(theta) + center[1]
    return x, y


def img2polar(img, center, final_radius,
              initial_radius = None, phase_width = 3000, step_radius=1):

    if initial_radius is None:
        initial_radius = 0
    theta , R = np.meshgrid(np.linspace(0, 2*np.pi,
                                        phase_width), 
                            np.arange(initial_radius,
                                      final_radius,
                                      step_radius))    
    len_tst = len(np.arange(initial_radius,
                            final_radius,
                            step_radius))
    Xcart, Ycart = polar2cart(R, theta, center)
    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)
    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(len_tst,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(len_tst,phase_width))
    return polar_img

def get_mask(polyno_array, pix_bi, xce, yce):
    darkening_polar = polyno_array
    darkening_cartesian = pol2cart(darkening_polar, (xce, yce))
    darkening_cartesian[darkening_cartesian==0]=4095#pow(2,16)
    return darkening_cartesian

def clv_correction(pixMat, radius, xce, yce, poly_order, pix_bit):

    pix_nb = pixMat.shape[0]
    
    zeroMat = np.zeros((3000,3000))
    zeroMat[:pix_nb, :pix_nb]= pixMat
    pol1 = img2polar(zeroMat,
                     (int(xce), int(yce)),
                     int(radius+100),
                     phase_width=720)
    pol1_inv = np.swapaxes(pol1, 0, 1)
    intensity_r = np.mean(pol1_inv, axis=0)
    radius_fit = radius 
    step_radius = 1
    radius_start = 0
    radius_end = int(radius_fit/step_radius)
    r_value = np.arange(0, radius_fit, step_radius)
    r_value = r_value * 1. /radius_fit
    mu = np.sqrt(1 - np.power(r_value, 2))
    
    x = mu[0:int(radius)]
    y = intensity_r[0:int(radius)]

    coefs, stats = P.polyfit(x, y, poly_order, full=True)    
    y_fit = np.polyval(coefs[::-1], x)
    polyno_array = np.zeros((pix_nb, pix_nb))
    polyno_array[:,0:int(radius)] = np.polyval(coefs[::-1], x)
    darkening_cartesian = get_mask(polyno_array, pix_bit, xce, yce)
    div = (pixMat / darkening_cartesian) 
    div_norm = ((div - div.min())  * pix_bit/(div.max() - div.min()))
    return div_norm


def draw_fits_image(fits_file, angle):
    hdulst = fits.open(fits_file)[0]
    uset_map = sunmap.GenericMap(hdulst.data, hdulst.header)

    rotated = uset_map.rotate(angle=-angle * u.deg)

    plt.figure()
    ax = plt.subplot(projection=rotated)        
    rotated.plot()
    ax.set_autoscale_on(True)

    plt.show()

def get_FITS_header(fits_file):
    hdulst = fits.open(fits_file)[0]
    uset_map = sunmap.GenericMap(hdulst.data, hdulst.header)
    return dict(hdulst.header)

def dump_FITS_image(fits_file, angle):
    hdulst = fits.open(fits_file)[0]
    uset_map = sunmap.GenericMap(hdulst.data, hdulst.header)

    # print(f'original: {uset_map.data.shape}')
    
    center = [hdulst.header['CENTER_X'],hdulst.header['CENTER_Y']]
    radius = [hdulst.header['CENTER_X'] + hdulst.header['SOLAR_R'],hdulst.header['CENTER_Y']]

    
    rotated = uset_map.rotate(angle=-angle * u.deg)

    offset = (rotated.data.shape[0] - uset_map.data.shape[0]) // 2
    # print(f'offset : {offset}')
    padding_width = (1 -( hdulst.header['SOLAR_R'] / (uset_map.data.shape[0]//2) ))/2
    # print(f'padding_width: {padding_width}')

    # print(f'rotated: {rotated.data.shape}')
    ### RESIZE ROTATED

    resized = rotated.data[offset:-offset, offset:-offset]
    plt.show()

    array = rotated.data

    r_img = rotate(array, 25, reshape=False)
    sk_img = skimage.transform.rotate(array,angle=25,resize=False, mode='constant')

    return np.flip(resized, axis=0), padding_width

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def parse_args():
    parser = argparse.ArgumentParser(description="Generate masks from Niels' method")


    parser.add_argument('--input_images_dir', dest='images_dir', required=True, type=str, 
                            help='Directory where the original natural images are stored')
    parser.add_argument('--output_dir', dest='output_dir', required=True, type=str, 
                            help='Directory where the output masks will be stored')

    args = parser.parse_args()

    return args

def main():
    import tqdm

    args = parse_args()
    images = sorted(glob.glob(os.path.join(args.images_dir, '*/*.FTS')))
    # images = []


    low = 100
    high = 140

    # print(images)

    for image in tqdm.tqdm(images[:]):

        basename = os.path.basename(image)
        base_no_ext = basename.split('.')[0]
        # print(basename)

        header = get_FITS_header(image)
        wl_resized, padding = dump_FITS_image(image, header['SOLAR_P0'])

        center = [header['CENTER_X'],header['CENTER_Y']]
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

        mask = create_circular_mask( wl_resized.shape[1], wl_resized.shape[0] ,center,radius*.95)

        # edges1 = filters.sobel(wl_resized, mask = mask)
        edges1 = filters.sobel(pixMat_flat, mask = mask)

        lowt = (edges1 > low).astype(int)
        hight = (edges1 > high).astype(int)
        hyst = filters.apply_hysteresis_threshold(edges1, low, high)

        vals = hight + hyst
        vals_unique = np.unique(vals)


        open_lowt = skimage.morphology.area_opening(lowt, area_threshold = 40)
        # print(np.unique(open_lowt))
        segmentation2 = watershed(edges1 , open_lowt+1)

        filled_mask2 = np.zeros_like(segmentation2)
        for i in np.unique(segmentation2)[1:]:
            seg = ndi.binary_fill_holes((segmentation2 == i))
            filled_mask2[seg ==1] = 255

        # print(np.unique(segmentation2))
        # print(np.unique(filled_mask2))
              

        # output filled_mask2
        cv2.imwrite(os.path.join(args.output_dir, base_no_ext+'.png'), filled_mask2)
        # PIL_image = Image.fromarray(filled_mask2)
        # PIL_image.save(os.path.join(args.output_dir, base_no_ext+'.png'))




if __name__ == "__main__":
    main()
