"""
Creates a shell from  aligned masks
"""
import argparse
import os
import sys
import numpy as np
import shutil
from skimage import io
from tqdm import tqdm
from atlas.NgSegmentMaker import NgConverter
from lib.utilities_cvat_neuroglancer import mask_to_shell
from lib.sqlcontroller import SqlController
from lib.file_location import FileLocationManager
from lib.utilities_process import test_dir, SCALING_FACTOR,get_max_imagze_size
from lib.utilities_create_alignment import parse_elastix, align_section_masks
from lib.utilities_mask import place_image,rotate_image
import tifffile as tiff
import cv2
from skimage import measure
import pickle as pk
from scipy.signal import savgol_filter
from lib.utilities_process import get_image_size
import scipy

def mask_to_shell(mask):
    sub_contours = measure.find_contours(mask, 1)

    sub_shells = []
    for sub_contour in sub_contours:
        sub_contour.T[[0, 1]] = sub_contour.T[[1, 0]]
        pts = sub_contour.astype(np.int32).reshape((-1, 1, 2))

        sub_shell = np.zeros(mask.shape, dtype='uint8')
        sub_shell = cv2.polylines(sub_shell, [pts], True, 1, 5, lineType=cv2.LINE_AA)
        sub_shells.append(sub_shell)
    shell = np.array(sub_shells).sum(axis=0)
    
    return shell

def align_masks(animal):
    rotate_and_pad_masks(animal)
    transforms = parse_elastix(animal)
    align_section_masks(animal,transforms)

def rotate_and_pad_masks(animal):
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    INPUT = fileLocationManager.thumbnail_mask
    rotation = sqlController.scan_run.rotation
    flip = sqlController.scan_run.flip
    OUTPUT = fileLocationManager.rotated_and_padded_thumbnail_mask
    os.makedirs(OUTPUT,exist_ok=True)
    max_width, max_height = get_max_imagze_size(fileLocationManager.get_thumbnail_aligned())
    for file in os.listdir(INPUT):
        infile = os.path.join(INPUT, file)
        outfile = os.path.join(OUTPUT, file)
        if os.path.exists(outfile):
            continue
        try:
            mask = io.imread(infile)
        except IOError as e:
            errno, strerror = e.args
            print(f'Could not open {infile} {errno} {strerror}')

        if rotation > 0:
            mask = rotate_image(mask, infile, rotation)
        if flip == 'flip':
            mask = np.flip(mask)
        if flip == 'flop':
            mask = np.flip(mask, axis=1)
        mask = place_image(mask, infile, max_width, max_height, 0)
        tiff.imwrite(outfile, mask)

def create_shell(animal, layer_type):
    '''
    Gets some info from the database used to create the numpy volume from
    the masks. It then turns that numpy volume into a neuroglancer precomputed
    mesh or image
    :param animal:
    :param layer_type:
    '''
    align_masks(animal)
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    INPUT = fileLocationManager.aligned_rotated_and_padded_thumbnail_mask
    error = test_dir(animal, INPUT, downsample=True, same_size=True)
    if len(error) > 0:
        print(error)
        sys.exit()

    if layer_type == 'image':
        dir = 'perimeter'
    else:
        dir = 'shell'
    OUTPUT_DIR = os.path.join(fileLocationManager.neuroglancer_data, dir)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(os.listdir(INPUT))
    volume = []
    for file in tqdm(files):
        tif = io.imread(os.path.join(INPUT, file))
        volume.append(tif)
    volume = np.array(volume).astype('uint8')
    volume = np.swapaxes(volume, 0, 2)
    volume = (volume!=0).astype('uint8')
    volume = scipy.ndimage.gaussian_filter(volume,sigma=3)
    ids = np.unique(volume)
    ids = [(i,i) for i in ids]
    resolution = sqlController.scan_run.resolution
    resolution = int(resolution * 1000 / SCALING_FACTOR)
    ng = NgConverter(volume, [resolution, resolution, 20000], offset=[0,0,0], layer_type=layer_type)
    if layer_type == 'segmentation':
        ng.create_neuroglancer_files(OUTPUT_DIR,ids)
    else:
        ng.create_neuroglancer_image(OUTPUT_DIR)




def create_shell_mask(animal):
    fileLocationManager = FileLocationManager(animal)
    INPUT = fileLocationManager.aligned_rotated_and_padded_thumbnail_mask
    files = os.listdir(INPUT)
    files = sorted(files)
    volume = []
    for filei in tqdm(files):
        img = io.imread(os.path.join(INPUT, filei))
        sub_shell = mask_to_shell(img)
        volume.append(sub_shell)

    create_volume(animal, volume)


def create_shell_thumbnail(animal):
    fileLocationManager = FileLocationManager(animal)
    INPUT = fileLocationManager.get_thumbnail_aligned()
    files = os.listdir(INPUT)
    files = sorted(files)
    volume = []
    for filei in tqdm(files):
        img = io.imread(os.path.join(INPUT, filei))
        mask = img>np.average(img)*0.5
        _,masks,stats,_=cv2.connectedComponentsWithStats(np.int8(mask))
        seg_sizes = stats[:,-1]
        second_largest = np.argsort(seg_sizes)[-2]
        mask = masks==second_largest
        sub_contours = measure.find_contours(mask, 0)
        sub_contour = sub_contours[0]
        sub_contour.T[[0, 1]] = sub_contour.T[[1, 0]]
        pts = sub_contour.astype(np.int32).reshape((-1, 2))
        if len(pts)>99:
            pts = savgol_filter((pts[:,0],pts[:,1]), 99, 1).T.astype(np.int32)
        sub_shell = np.zeros(mask.shape, dtype='uint8')
        sub_shell = cv2.polylines(sub_shell, [pts], True, 1, 5, lineType=cv2.LINE_AA)
        volume.append(sub_shell)

    create_volume(animal, volume)

def create_volume(animal, volume):
    fileLocationManager = FileLocationManager(animal)
    OUTPUT_DIR = os.path.join(fileLocationManager.neuroglancer_data, 'shell')
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sqlController = SqlController(animal)
    volume = np.array(volume).astype('uint8')
    volume = np.swapaxes(volume, 0, 2)
    volume = (volume!=0).astype('uint8')
    ids = np.unique(volume)
    ids = [(i,i) for i in ids]
    resolution = sqlController.scan_run.resolution
    resolution = int(resolution * 1000 / SCALING_FACTOR)
    ng = NgConverter(volume, [resolution, resolution, 20000], offset=[0,0,0])
    ng.create_neuroglancer_files(OUTPUT_DIR, ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--layer_type', help='Enter the layer_type: segmentation|image', required=True)
    args = parser.parse_args()
    animal = args.animal
    layer_type = args.layer_type.lower()
    layers = ['segmentation', 'image']
    if layer_type not in layers:
        print(f'Layer type is incorrect {layer_type}')
        sys.exit()
    create_shell_thumbnail(animal)

