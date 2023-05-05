from numpy.lib.function_base import interp
from sunscc.dataset.transform.pipelines import Compose
import collections
import zipfile
from functools import partial
from torch.utils.data import Dataset
from pathlib import Path
from cachetools import cached
from cachetools.keys import hashkey
from collections import namedtuple, defaultdict
from skimage.util import view_as_windows
import nibabel
import logging
import numpy as np
import cv2
from hydra.utils import call, instantiate
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def read_sample(samplefiles=None, **kwargs):
    if samplefiles is None:
        samplefiles = kwargs
    else:
        samplefiles.update(kwargs)

    result = {}
    for part in samplefiles:
        filepath = Path(samplefiles[part])
        part_type = filepath.suffixes

        if ".nii" in part_type:
            result[part] = nibabel.load(filepath).get_fdata()
        else:
            raise ValueError(
                f"{part} is either not a valid type or not yet implemented."
            )

    return result


class DirectoryDataset(Dataset):
    def __init__(self, root_dir, dtypes):
        self.dtypes = dtypes
        self.fnames = {}
        for i, dtype in enumerate(dtypes):
            fnames = (Path(root_dir) / dtype).iterdir()
            self.fnames[dtype] = sorted(fnames)

        all_len = [len(elem) for elem in self.fnames.values()]

        # Check if all directories have the same number of files
        assert (
            len(np.unique(all_len)) == 1
        ), f"The number of images is not equal for the different inputs:\n {dtypes}\n {all_len}"

        self.length = all_len[0]
        self.fname_samples = [
            dict(zip(self.fnames, i)) for i in zip(*self.fnames.values())
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        result = read_sample(self.fname_samples[idx])
        return result


class MultipleNumpyDataset(Dataset):
    """ A dataset containing a directory per type (image, segmentation)

        Every type directory contains an .npz file per sample (or volume), the 
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
        self, root_dir, partition, dtypes, transforms=None, remove_start=0, remove_end=0
    ) -> None:
        super().__init__()
        if isinstance(transforms, collections.Mapping):
            transforms = partial(call, config=transforms)
        elif isinstance(transforms, collections.Sequence):
            transforms_init = []
            for transform in transforms:
                transforms_init.append(instantiate(transform))
            transforms = Compose(transforms_init)
        self.transforms = transforms
        self.root_dir = Path(root_dir) / partition
        self.dtypes = dtypes
        self.data = None
        self.files = []
        self.reverse_index = None
        self.array_index = None
        self.original_shape = None
        self.remove_start = remove_start
        self.remove_end = remove_end
        self.main_dtype = dtypes[0]
        all_len = []
        for dtype in dtypes:
            dtype_len = 0
            for file in sorted((self.root_dir / dtype).iterdir()):
                if dtype == self.main_dtype:
                    self.files.append(file.name)
                data = np.load(file)
                dtype_len += len(
                    data.files[self.remove_start : len(data) - self.remove_end]
                )
            all_len.append(dtype_len)

        log.debug(f"{self.files}")
        if len(np.unique(all_len)) != 1:
            log.warning("Unequal number of images")

    def __len__(self) -> int:
        length = 0
        for file in self.files:
            data = np.load(self.root_dir / self.main_dtype / file)
            length += len(data.files[self.remove_start : len(data) - self.remove_end])
        return length

    def initialize(self):
        self.reverse_index = defaultdict(list)
        self.array_index = defaultdict(list)
        self.data = defaultdict(list)
        self.original_shape = defaultdict(list)
        for dtype in self.dtypes:
            for i, filename in enumerate(self.files):
                file = self.root_dir / dtype / filename
                try:
                    data = np.load(file)
                except Exception as e:
                    # log.debug(e)
                    continue
                for j in range(self.remove_start, len(data) - self.remove_end):
                    self.reverse_index[dtype].append(i)
                    self.array_index[dtype].append(j)
                self.data[dtype].append(data)
                self.original_shape[dtype].append(data[data.files[0]].shape)

    def reset(self):
        for _, data in self.data.items():
            for item in data:
                item.close()

        self.data = None
        self.reverse_index = None
        self.array_index = None

    def __getitem__(self, index: int, do_transform=True):
        if self.data is None:
            self.initialize()

        sample = {}
        log.debug(
            "%s, idx: %d; file: %d; array: %d",
            self.main_dtype,
            index,
            self.reverse_index["image"][index],
            self.array_index["image"][index],
        )
        file_index = self.reverse_index[self.main_dtype][index]
        array_index = self.array_index[self.main_dtype][index]
        array_name = self.data[self.main_dtype][file_index].files[array_index]
        for dtype in self.dtypes:
            try:
                sample[dtype] = self.data[dtype][file_index][array_name]
            except Exception as e:
                log.debug(f"{dtype} {file_index} {array_name}")
                log.debug(f"{e}")

        if self.transforms is not None and do_transform:
            sample = self.transforms(**sample)
        return sample


class NumpyDataset(Dataset):
    def __init__(self, root_dir, dtypes):
        self.root_dir = root_dir
        self.dtypes = dtypes
        self.files = None
        all_len = []
        for dtype in dtypes:
            data = np.load(root_dir / (dtype + ".npz"))
            # self.files[dtype] = data
            all_len.append(len(data.files))

        assert len(np.unique(all_len)) == 1, f"unequal number of images"

        self.length = all_len[0]

    def __len__(self) -> int:
        return self.length

    def initialize(self):
        self.files = {}
        for dtype in self.dtypes:
            data = np.load(self.root_dir / (dtype + ".npz"))
            self.files[dtype] = data

    def __getitem__(self, index: int):
        # initialize file pointers inside each individual worker
        if self.files is None:
            self.initialize()
        return {
            dtype: self.files[dtype][self.files[dtype].files[index]]
            for dtype in self.dtypes
        }

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_sun_maskOLD(img, circular_mask,  sun_radius=100):
    '''Get the mask of the sun from the image using whitehat transform'''
    # image is single channel, convert to 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_shape = img.shape
    
    img_scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_blur = cv2.GaussianBlur(img_scaled, (5,5), 0) 

    sun_mask = cv2.Canny(image=img_scaled, threshold1=40, threshold2=40) # Canny Edge Detection
    
    circles = cv2.HoughCircles(sun_mask, cv2.HOUGH_GRADIENT, 1, 1000, param1=50, param2=30, minRadius=int(img_shape[0]//4), maxRadius=int(1.1*sun_radius))
    circles = np.uint16(np.around(circles))
    # print(circles)
    tmp = np.zeros_like(sun_mask)
    for i in circles[0,:]:
        dist_to_center = np.sqrt((i[0] - img_shape[0]//2)**2 + (i[1] - img_shape[1]//2)**2)
        if dist_to_center < sun_radius//4:
            cv2.circle(tmp,(i[0],i[1]),i[2],(255,255,255),1)
            
    sun_mask = tmp + sun_mask

    # borderType = cv2.BORDER_REPLICATE
    borderType = cv2.BORDER_CONSTANT

    # do a dilation operation to fill the gaps between the edges
    sun_mask = cv2.dilate(sun_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), borderType=borderType, borderValue=0)
    sun_mask = sun_mask * circular_mask

    after_dilate = sun_mask.copy()

    # do a closing operation to close the gaps between the edges
    sun_mask = cv2.morphologyEx(sun_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    
    after_close1 = sun_mask.copy()

    im_floodfill = sun_mask.copy()
    h, w = sun_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (h//2,w//2), 255)
    # pick points on a circle around the center of the image
    num_points = 20
    circle_points = np.linspace(0, 2*np.pi, num_points)
    circle_radius = .9*sun_radius if sun_radius < min(h,w)//2 else .5*min(h,w)//2
    pts = []
    for i in range(num_points):
        x = int(h//2 + circle_radius*np.cos(circle_points[i]))
        y = int(w//2 + circle_radius*np.sin(circle_points[i]))
        pts.append((x,y))
        cv2.floodFill(im_floodfill, mask, (x,y), 255)

   

    im_out = im_floodfill
   
    im_out = cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)



    result = im_out
    r2 = im_out.copy()
    # do a closing operation to close the holes in the sun that may have been created by the erosion
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    after_close2 = result.copy()

    result = ((r2 + result)>0).astype(np.uint8)*255
    # # do an erosion operation to remove the edges of the sun
    result = cv2.erode(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    after_erode = result.copy()


    result[img[:,:,0] == 0] = 0
    # convert to grayscale
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = result * circular_mask

    return result #, sun_mask, pts , after_dilate, after_close1, r2, after_close2, after_erode


def get_blob_mask(binary_mask, point):
    # Get the connected components in the binary mask
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Get the coordinates of the given point
    x, y = point

    # Loop through each connected component and check if the given point is inside it
    for i in range(1, nb_components):
        # Get the bounding box of the connected component
        x1, y1, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x2, y2 = x1 + w, y1 + h
        
        # Check if the given point is inside the bounding box
        if x1 <= x <= x2 and y1 <= y <= y2:
            # Create a mask for the connected component
            blob_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            blob_mask[output == i] = 1

            return blob_mask

def get_sun_mask(img, circular_mask,  sun_radius=100):
    '''Get the mask of the sun from the image using whitehat transform'''
    # image is single channel, convert to 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_shape = img.shape
    # print(img_shape)

    img_scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    img_scaled_th = img_scaled>40
    img_scaled_th = img_scaled_th[:,:,0]
    img_scaled_th = img_scaled_th.astype(np.uint8)*255

    # print(img_scaled_th.shape)

    result1 = img_scaled_th.copy()
    #do an erosion operation to remove the edges of the sun
    result1 = cv2.erode(result1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    result1 = get_blob_mask(result1, (img_shape[1]//2, img_shape[0]//2))

    #################

    sun_mask = cv2.Canny(image=img_scaled, threshold1=40, threshold2=40) # Canny Edge Detection


    borderType = cv2.BORDER_CONSTANT
    sun_mask = cv2.dilate(sun_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), borderType=borderType, borderValue=0)
    sun_mask = sun_mask * circular_mask
    
    sun_mask = cv2.morphologyEx(sun_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    

    im_floodfill = sun_mask.copy()
    h, w = sun_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (h//2,w//2), 255)
    
    num_points = 20
    circle_points = np.linspace(0, 2*np.pi, num_points)
    circle_radius = .9*sun_radius if sun_radius < min(h,w)//2 else .5*min(h,w)//2
    pts = []
    for i in range(num_points):
        x = int(h//2 + circle_radius*np.cos(circle_points[i]))
        y = int(w//2 + circle_radius*np.sin(circle_points[i]))
        pts.append((x,y))
        cv2.floodFill(im_floodfill, mask, (x,y), 255)


    result2 = im_floodfill.copy()


    pad_width = 50
    result2= cv2.copyMakeBorder(result2,pad_width,pad_width,pad_width,pad_width,cv2.BORDER_CONSTANT,value=0)
    result1= cv2.copyMakeBorder(result1,pad_width,pad_width,pad_width,pad_width,cv2.BORDER_CONSTANT,value=0)

    result2 = cv2.morphologyEx(result2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_width, pad_width)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    resul2 = cv2.erode(result2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    result1 = cv2.morphologyEx(result1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_width, pad_width)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # don't erode result1, it is normal

    result2 = result2[pad_width:-pad_width, pad_width:-pad_width]
    result1 = result1[pad_width:-pad_width, pad_width:-pad_width]


    result = result2*result1

    return result


def create_excentricity_map(mask, sun_radius, value_outside=np.nan):
    """
    Create a map of the excentricity of the sun in the image.
    """
    # get the center of the sun
    center = np.array(mask.shape)//2
    # create a meshgrid of the same size as the image
    x, y = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
    # compute the distance between each pixel and the center of the sun
    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)

    ratio = dist/sun_radius
    ratio = np.clip(ratio, -1, 1)
    rho = np.arcsin(ratio)
    rho_deg = np.rad2deg(rho)

    rho_deg[rho_deg>=90] = value_outside

    # compute the excentricity map
    excentricity_map = rho_deg
    return excentricity_map

def whitelight_to_datetime(name):
    name2= name
    name2 = name2.replace('UPH','')

    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]
    seconds = name2[12:14]

    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes, "seconds": seconds}


def datetime_to_db_string(datetime):
    date = '-'.join([datetime['year'],datetime['month'], datetime['day']])
    time = ':'.join([datetime['hours'], datetime['minutes'], '00'])
    
    return f'{date} {time}'
