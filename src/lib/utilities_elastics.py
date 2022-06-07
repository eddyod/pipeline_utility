import os
from subprocess import check_output
import numpy as np
from lib.utilities_registration import register_simple
from lib.sqlcontroller import SqlController


def create_elastix(animal):

    DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    INPUT = os.path.join(DIR, 'CH1', 'thumbnail_cleaned')
    sqlController = SqlController(animal)
    files = sorted(os.listdir(INPUT))
    for i in range(1, len(files)):
        fixed_index = os.path.splitext(files[i-1])[0]
        moving_index = os.path.splitext(files[i])[0]
        if not sqlController.check_elastix_row(animal, moving_index):
            rotation, xshift, yshift = register_simple(
                INPUT, fixed_index, moving_index)
            sqlController.add_elastix_row(
                animal, moving_index, rotation, xshift, yshift)


def create_within_stack_transformations(animal):
    """Calculate and store the rigid transformation using elastix.  The transformations are calculated from the next image to the previous
    """
    debug = False
    sqlController = SqlController(animal)
    DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    INPUT = os.path.join(DIR, 'CH1', 'thumbnail_cleaned')
    files = sorted(os.listdir(INPUT))
    for i in range(1, len(files)):
        fixed_index = os.path.splitext(files[i-1])[0]
        moving_index = os.path.splitext(files[i])[0]
        if not sqlController.check_elastix_row(animal, moving_index):
            second_transform_parameters, initial_transform_parameters = \
                register_simple(INPUT, fixed_index, moving_index, debug)
            T1 = parameters_to_rigid_transform(*initial_transform_parameters)
            T2 = parameters_to_rigid_transform(
                *second_transform_parameters, get_rotation_center())
            T = T1@T2
            xshift, yshift, rotation, _ = rigid_transform_to_parmeters(
                animal, T)
            sqlController.add_elastix_row(
                animal, moving_index, rotation, xshift, yshift)


def rigid_transform_to_parmeters(animal, transform):
    """convert a 2d transformation matrix (3*3) to the rotation angles, rotation center and translation

    Args:
        transform (array like): 3*3 array that stores the 2*2 transformation matrix and the 1*2 translation vector for a 
        2D image.  the third row of the array is a place holder of values [0,0,1].

    Returns:
        float: x translation
        float: y translation
        float: rotation angle in arc
        list:  lisf of x and y for rotation center
    """
    R = transform[:2, :2]
    shift = transform[:2, 2]
    tan = R[1, 0]/R[0, 0]
    center = get_rotation_center(animal)
    rotation = np.arctan(tan)
    xshift, yshift = shift-center + np.dot(R, center)
    return xshift, yshift, rotation, center


def parameters_to_rigid_transform(rotation, xshift, yshift, center):
    """convert a set of rotation parameters to the transformation matrix

    Args:
        rotation (float): rotation angle in arc
        xshift (float): translation in x
        yshift (float): translation in y
        center (list): list of x and y for the rotation center

    Returns:
        array: 3*3 transformation matrix for 2D image, contain the 2*2 array and 1*2 translation vector
    """
    return parameters_to_rigid_transform(rotation, xshift, yshift, center)


def get_rotation_center(animal):
    """return a rotation center for finding the parameters of a transformation from the transformation matrix

    Returns:
        list: list of x and y for rotation center that set as the midpoint of the section that is in the middle of the stack
    """
    DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    INPUT = os.path.join(DIR, 'CH1', 'thumbnail_cleaned')
    files = sorted(os.listdir(INPUT))
    midpoint = len(files) // 2
    midfilepath = os.path.join(INPUT, files[midpoint])
    width, height = get_image_size(midfilepath)
    center = np.array([width, height]) / 2
    return center


def get_image_size(filepath):
    result_parts = str(check_output(["identify", filepath]))
    results = result_parts.split()
    width, height = results[2].split('x')
    return int(width), int(height)
