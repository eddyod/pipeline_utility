import os
import numpy as np
from skimage import io

animal = 'DK78'
DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
INPUT = os.path.join(DIR, 'CH1', 'thumbnail_cleaned')

badfile = '181.tif'
badpath = os.path.join(INPUT, badfile)
badarr = io.imread(badpath)
print(badfile,badarr.dtype, badarr.shape)

io.imsave(badpath, badarr)