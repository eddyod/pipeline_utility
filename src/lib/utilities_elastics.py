import os
from subprocess import check_output
import numpy as np
#from lib.utilities_registration import register_simple
from lib.sqlcontroller import SqlController
import SimpleITK as sitk


def create_elastixXXX(animal):

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


def align_elastix(fixed, moving, debug, tries=10):
    for _ in range(tries):
        try:
            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.SetFixedImage(fixed)
            elastixImageFilter.SetMovingImage(moving)
            rigid_params = elastixImageFilter.GetDefaultParameterMap("rigid")

            rigid_params["AutomaticTransformInitializationMethod"] = [
                "GeometricalCenter"
            ]
            rigid_params["ShowExactMetricValue"] = ["false"]
            rigid_params["CheckNumberOfSamples"] = ["true"]
            rigid_params["NumberOfSpatialSamples"] = ["5000"]
            rigid_params["SubtractMean"] = ["true"]
            rigid_params["MaximumNumberOfSamplingAttempts"] = ["0"]
            rigid_params["SigmoidInitialTime"] = ["0"]
            rigid_params["MaxBandCovSize"] = ["192"]
            rigid_params["NumberOfBandStructureSamples"] = ["10"]
            rigid_params["UseAdaptiveStepSizes"] = ["true"]
            rigid_params["AutomaticParameterEstimation"] = ["true"]
            rigid_params["MaximumStepLength"] = ["10"]
            rigid_params["NumberOfGradientMeasurements"] = ["0"]
            rigid_params["NumberOfJacobianMeasurements"] = ["1000"]
            rigid_params["NumberOfSamplesForExactGradient"] = ["100000"]
            rigid_params["SigmoidScaleFactor"] = ["0.1"]
            rigid_params["ASGDParameterEstimationMethod"] = ["Original"]
            rigid_params["UseMultiThreadingForMetrics"] = ["true"]
            rigid_params["SP_A"] = ["20"]
            rigid_params["UseConstantStep"] = ["false"]
            ## The internal pixel type, used for internal computations
            ## Leave to float in general.
            ## NB: this is not the type of the input images! The pixel
            ## type of the input images is automatically read from the
            ## images themselves.
            ## This setting can be changed to "short" to save some memory
            ## in case of very large 3D images.
            rigid_params["FixedInternalImagePixelType"] = ["float"]
            rigid_params["MovingInternalImagePixelType"] = ["float"]
            ## note that some other settings may have to specified
            ## for each dimension separately.
            rigid_params["FixedImageDimension"] = ["2"]
            rigid_params["MovingImageDimension"] = ["2"]
            ## Specify whether you want to take into account the so-called
            ## direction cosines of the images. Recommended: true.
            ## In some cases, the direction cosines of the image are corrupt,
            ## due to image format conversions for example. In that case, you
            ## may want to set this option to "false".
            rigid_params["UseDirectionCosines"] = ["true"]
            ## **************** Main Components **************************
            ## The following components should usually be left as they are:
            rigid_params["Registration"] = ["MultiResolutionRegistration"]
            rigid_params["Interpolator"] = ["BSplineInterpolator"]
            rigid_params["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
            rigid_params["Resampler"] = ["DefaultResampler"]
            ## These may be changed to Fixed/MovingSmoothingImagePyramid.
            ## See the manual.
            ##(FixedImagePyramid "FixedRecursiveImagePyramid']
            ##(MovingImagePyramid "MovingRecursiveImagePyramid']
            rigid_params["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
            rigid_params["MovingImagePyramid"] = [
                "MovingSmoothingImagePyramid"
            ]
            ## The following components are most important:
            ## The optimizer AdaptiveStochasticGradientDescent (ASGD) works
            ## quite ok in general. The Transform and Metric are important
            ## and need to be chosen careful for each application. See manual.
            rigid_params["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
            rigid_params["Transform"] = ["EulerTransform"]
            ##(Metric "AdvancedMattesMutualInformation")
            ## testing 17 dec
            rigid_params["Metric"] = ["AdvancedNormalizedCorrelation"]
            ## ***************** Transformation **************************
            ## Scales the rotations compared to the translations, to make
            ## sure they are in the same range. In general, it's best to
            ## use automatic scales estimation:
            rigid_params["AutomaticScalesEstimation"] = ["true"]
            ## Automatically guess an initial translation by aligning the
            ## geometric centers of the fixed and moving.
            rigid_params["AutomaticTransformInitialization"] = ["true"]
            ## Whether transforms are combined by composition or by addition.
            ## In generally, Compose is the best option in most cases.
            ## It does not influence the results very much.
            rigid_params["HowToCombineTransforms"] = ["Compose"]
            ## ******************* Similarity measure *********************
            ## Number of grey level bins in each resolution level,
            ## for the mutual information. 16 or 32 usually works fine.
            ## You could also employ a hierarchical strategy:
            ##(NumberOfHistogramBins 16 32 64)
            rigid_params["NumberOfHistogramBins"] = ["32"]
            ## If you use a mask, this option is important.
            ## If the mask serves as region of interest, set it to false.
            ## If the mask indicates which pixels are valid, then set it to true.
            ## If you do not use a mask, the option doesn't matter.
            rigid_params["ErodeMask"] = ["false"]
            ## ******************** Multiresolution **********************
            ## The number of resolutions. 1 Is only enough if the expected
            ## deformations are small. 3 or 4 mostly works fine. For large
            ## images and large deformations, 5 or 6 may even be useful.
            rigid_params["NumberOfResolutions"] = ["6"]
            ##(FinalGridSpacingInVoxels 8.0 8.0)
            ##(GridSpacingSchedule 6.0 6.0 4.0 4.0 2.5 2.5 1.0 1.0)
            ## The downsampling/blurring factors for the image pyramids.
            ## By default, the images are downsampled by a factor of 2
            ## compared to the next resolution.
            ## So, in 2D, with 4 resolutions, the following schedule is used:
            ##(ImagePyramidSchedule 4 4  2 2  1 1 )
            ## And in 3D:
            ##(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 )
            ## You can specify any schedule, for example:
            ##(ImagePyramidSchedule 4 4  4 3  2 1  1 1 )
            ## Make sure that the number of elements equals the number
            ## of resolutions times the image dimension.
            ## ******************* Optimizer ****************************
            ## Maximum number of iterations in each resolution level:
            ## 200-500 works usually fine for rigid registration.
            ## For more robustness, you may increase this to 1000-2000.
            ## 80 good results, 7 minutes on basalis with 4 jobs
            ## 200 good results except for 1st couple were not aligned, 12 minutes
            ## 500 is best, including first sections, basalis took 21 minutes
            rigid_params["MaximumNumberOfIterations"] = ["1200"]
            ## The step size of the optimizer, in mm. By default the voxel size is used.
            ## which usually works well. In case of unusual high-resolution images
            ## (eg histology) it is necessary to increase this value a bit, to the size
            ## of the "smallest visible structure" in the image:
            ##(MaximumStepLength 4)
            ## **************** Image sampling **********************
            ## Number of spatial samples used to compute the mutual
            ## information (and its derivative) in each iteration.
            ## With an AdaptiveStochasticGradientDescent optimizer,
            ## in combination with the two options below, around 2000
            ## samples may already suffice.
            ##(NumberOfSpatialSamples 2048)
            ## Refresh these spatial samples in every iteration, and select
            ## them randomly. See the manual for information on other sampling
            ## strategies.
            rigid_params["NewSamplesEveryIteration"] = ["true"]
            rigid_params["ImageSampler"] = ["Random"]
            ## ************* Interpolation and Resampling ****************
            ## Order of B-Spline interpolation used during registration/optimisation.
            ## It may improve accuracy if you set this to 3. Never use 0.
            ## An order of 1 gives linear interpolation. This is in most
            ## applications a good choice.
            rigid_params["BSplineInterpolationOrder"] = ["1"]
            ## Order of B-Spline interpolation used for applying the final
            ## deformation.
            ## 3 gives good accuracy; recommended in most cases.
            ## 1 gives worse accuracy (linear interpolation)
            ## 0 gives worst accuracy, but is appropriate for binary images
            ## (masks, segmentations); equivalent to nearest neighbor interpolation.
            rigid_params["FinalBSplineInterpolationOrder"] = ["3"]
            ##Default pixel value for pixels that come from outside the picture:
            rigid_params["DefaultPixelValue"] = ["0"]
            ## Choose whether to generate the deformed moving image.
            ## You can save some time by setting this to false, if you are
            ## only interested in the final (nonrigidly) deformed moving image
            ## for example.
            rigid_params["WriteResultImage"] = ["false"]
            ## The pixel type and format of the resulting deformed moving image
            rigid_params["ResultImagePixelType"] = ["unsigned char"]
            rigid_params["ResultImageFormat"] = ["tif"]
            rigid_params["RequiredRatioOfValidSamples"] = ["0.05"]
            elastixImageFilter.SetParameterMap(rigid_params)
            if debug:
                elastixImageFilter.LogToConsoleOn()
            else:
                elastixImageFilter.LogToConsoleOff()
        except RuntimeError:
            continue
        break
    elastixImageFilter.Execute()
    return (
        elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"])


def create_elastix(animal):
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
            second_transform_parameters, initial_transform_parameters = register_simple(INPUT, fixed_index, moving_index, debug)
            T1 = parameters_to_rigid_transform(*initial_transform_parameters)
            T2 = parameters_to_rigid_transform(
                *second_transform_parameters, get_rotation_center(animal))
            T = T1@T2
            xshift, yshift, rotation, _ = rigid_transform_to_parmeters(
                animal, T)
            sqlController.add_elastix_row(
                animal, moving_index, rotation, xshift, yshift)


def register_simple(INPUT, fixed_index, moving_index, debug=False):
    pixelType = sitk.sitkFloat32
    fixed_file = os.path.join(INPUT, f"{fixed_index}.tif")
    moving_file = os.path.join(INPUT, f"{moving_index}.tif")
    fixed = sitk.ReadImage(fixed_file, pixelType)
    moving = sitk.ReadImage(moving_file, pixelType)
    initial_transform = sitk.Euler2DTransform()
    initial_transform = parse_sitk_rigid_transform(initial_transform)
    # moving,initial_transform = align_principle_axis(moving,fixed)
    elastix_transform = align_elastix(fixed, moving, debug=debug)
    return (elastix_transform, initial_transform)


def parse_sitk_rigid_transform(sitk_rigid_transform):
    rotation, xshift, yshift = sitk_rigid_transform.GetParameters()
    center = sitk_rigid_transform.GetFixedParameters()
    return rotation, xshift, yshift, center


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

    def rigid_transform(rotation, xshift, yshift, center):
        rotation, xshift, yshift = np.array([rotation, xshift, yshift]).astype(
            np.float16
        )
        center = np.array(center).astype(np.float16)
        R = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        shift = center + (xshift, yshift) - np.dot(R, center)
        T = np.vstack([np.column_stack([R, shift]), [0, 0, 1]])
        return T



    return rigid_transform(rotation, xshift, yshift, center)


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
