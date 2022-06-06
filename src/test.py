import os
import numpy as np
from skimage import io

INPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DKEA/preps/CH1/thumbnail_aligned'


files = sorted(os.listdir(INPUT))

len_files = len(files)
midpoint = len_files // 2
midfilepath = os.path.join(INPUT, files[midpoint])
midfile = io.imread(midfilepath)
data_type = midfile.dtype
#image = np.load('/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/structures/allen/allen.npy')
ids = np.unique(midfile)
print(type(ids))
print(ids)
l = [str(value) for value in ids.tolist()]
print(l)