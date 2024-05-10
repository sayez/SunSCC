from create_detection_masks import *
import glob
import cv2
from skimage import data, filters, segmentation

from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
from skimage.measure import shannon_entropy
from skimage.morphology import square, disk
import numpy as np
from skimage.measure import label, regionprops


def search_max_threshold(region, tmp2):
    min_th = 1000
    
    thresh_history = []
    history_pen = []
    history_um = []
    init_max_th = 2000
    for i in range(30):
        tmp3 = None
        ok = False
        while not ok or init_max_th > np.max(tmp2) :
            max_th = init_max_th + i*100
    #         thresholds = filters.threshold_multiotsu(region[np.where((region < max_th) & (region > min_th))], classes = 3)     
            tmp3 = tmp2[np.where((tmp2 <= max_th) & (tmp2 >= min_th))]
#             print(np.unique(tmp3).shape)
            if len(np.unique(tmp3)) < 3:
                init_max_th += 100
            else:
                ok = True
#                 print(max_th)
        thresholds = filters.threshold_multiotsu(tmp3, classes = 3)     
        levels = np.digitize(tmp2, bins=thresholds)
        
        
        tmp_penumbrae = segmentation.clear_border(levels == 1)
        tmp_umbrae = segmentation.clear_border(levels == 0)
        
        label_pen = label(tmp_penumbrae)
        label_um = label(tmp_umbrae)

        pen_props = regionprops(label_pen)
        um_props = regionprops(label_um)
        
#         print(f'{len(um_props)}, {len(pen_props)}')
        
        if len(history_um) == 0 and len(history_pen) == 0:    
            history_pen.append(len(pen_props))
            history_um.append(len(um_props))
            thresh_history.append(thresholds)
            continue
        
        if (((len(history_um) > 0) and (len(um_props) - history_um[0] < 4  )) and
            ((len(history_pen) > 0) and (len(pen_props) - history_pen[-1] < 10  ))):
            #stop
            history_pen.append(len(pen_props))
            history_um.append(len(um_props))
            thresh_history.append(thresholds) 
        else:
            return thresh_history[-1]
 
    return thresh_history[-1]

def local_hist(image, center, offset):
    # init_region = image[center[1]-padding : center[1]+padding , center[0]-padding : center[0]+padding]
    img_cpy= image.copy()

    # print(center[1]-offset)
    # print(center[1]+offset)
    # print(center[0]-offset)
    # print(center[0]-offset)
    
    region = img_cpy[center[1]-offset : center[1]+offset , center[0]-offset : center[0]+offset]
    
    tmp = outside[center[1]-offset : center[1]+offset , center[0]-offset : center[0]+offset]
    tmp2 = np.stack((tmp[:,:,None], region[:,:, None]), axis =-1)
    tmp2 = np.max(tmp2, axis=-1).squeeze()
    
    thresholds = search_max_threshold(region, tmp2)
    levels = np.digitize(tmp2, bins=thresholds)
    
    levels = 2- levels
    levels = skimage.morphology.area_opening(levels, area_threshold = 10)
    levels = skimage.morphology.opening(levels,disk(1))
    # ent = shannon_entropy(levels)

    return levels
   

################################  SCRIPT  ##########################################################

images_dir = '/home/sayez/DATASETS/USET/USET_images/USET_White_Light/2013_L1c/2013'
output_dir = '/home/sayez/DATASETS/Otsu_local'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

offset = 50

images = sorted(glob.glob(os.path.join(images_dir, '*/*.FTS')))

for image in tqdm.tqdm(images[:10]):
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

    # mask = create_circular_mask( wl_resized.shape[1], wl_resized.shape[0] ,center,radius*.99)
    mask = create_circular_mask( wl_resized.shape[1], wl_resized.shape[0] ,center,radius*0.96)

    outside = pixMat_flat.copy()#np.max(pixMat_flat) * (1-mask)
    outside[outside < 900] = np.max(outside)
    outside[outside != np.max(outside)] = 0

    # Threshold the image first then clear the border
    im_penumbrae = segmentation.clear_border(pixMat_flat < 3100)
    im_umbrae = segmentation.clear_border(pixMat_flat < 2500)

    label_im_penumbrae = label(im_penumbrae)
    label_im_umbrae = label(im_umbrae)

    props_umbrae = regionprops(label_im_umbrae)
    props_penumbrae = regionprops(label_im_penumbrae)

    penumbrae_props_centers = []
    for prop in props_penumbrae:
        if prop.area > 15:
            center = np.array(prop.centroid)
            center = np.array([round(center[1]), round(center[0])])
            penumbrae_props_centers.append(center)
            # print(f'area: {prop.area} ; center : {center}')

    umbrae_props_centers = []
    for prop in props_umbrae:
        if prop.area > 15:
            center = np.array(prop.centroid)
            center = np.array([round(center[1]), round(center[0])])
            umbrae_props_centers.append(center)    

    # Find contours and compute region centers of umbrae
    contours, hierarchy = cv2.findContours(im_umbrae.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centers_umbrae = []
    for contour in contours:
        center = None
        if contour.shape[0] < 5 :
            continue
        else:
            M = cv2.moments(contour)
            if M['m00'] == 0 :
                continue
            center = np.array([round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])])
            
        centers_umbrae.append(center)


    # Find contours and compute region centers of penumbrae
    contours, hierarchy = cv2.findContours(im_penumbrae.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centers_penumbrae = []
    for contour in contours:

        center = None
        if contour.shape[0] < 5 :
            continue
        else:
            M = cv2.moments(contour)
            if M['m00'] == 0 :
                continue
            center = np.array([round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])])
            
        centers_penumbrae.append(center)

    # create empty mask
    out_mask = np.zeros_like(pixMat_flat)

    # plt.figure()
    # plt.imshow(out_mask)

    for center in penumbrae_props_centers:
        print(center)
        print(offset)
        levels = local_hist(pixMat_flat, center, offset)
        # print(levels.shape)
        out_mask[center[1]-offset:center[1]+offset, center[0]-offset:center[0]+offset] = levels

    
    cv2.imwrite(os.path.join(output_dir, base_no_ext+'.png'), out_mask)

