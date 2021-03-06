"""
Creates a shell from  aligned thumbnails
"""
import argparse
import os
import sys
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
HOME = os.path.expanduser("~")
#PATH = os.path.join(HOME, 'programming/pipeline_utility')
#sys.path.append(PATH)
from lib.file_location import FileLocationManager
from lib.utilities_cvat_neuroglancer import calculate_chunks, calculate_factors
from lib.utilities_process import get_cpus
def create_downsamples(animal, channel, suffix, downsample,njobs):
    fileLocationManager = FileLocationManager(animal)
    channel_outdir = f'C{channel}'
    first_chunk = calculate_chunks(downsample, 0)
    mips = [0,1,2,3,4,5,6,7]

    if downsample:
        channel_outdir += 'T'
        mips = [0,1]
 

    outpath = os.path.join(fileLocationManager.neuroglancer_data, f'{channel_outdir}')
    outpath = f'file://{outpath}'
    if suffix is not None:
        outpath += suffix

    channel_outdir += "_rechunkme"
    INPUT_DIR = os.path.join(fileLocationManager.neuroglancer_data, f'{channel_outdir}')

    if not os.path.exists(INPUT_DIR):
        print(f'DIR {INPUT_DIR} does not exist, exiting.')
        sys.exit()

    cloudpath = f"file://{INPUT_DIR}"
    # _, workers = get_cpus()
    tq = LocalTaskQueue(parallel=njobs)

    tasks = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, 
        chunk_size=first_chunk, mip=0, skip_downsamples=True)
    tq.insert(tasks)
    tq.execute()

    #mips = 7 shows good results in neuroglancer
    for mip in mips:
        cv = CloudVolume(outpath, mip)
        chunks = calculate_chunks(downsample, mip)
        factors = calculate_factors(downsample, mip)
        tasks = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, factor=factors, preserve_chunk_size=False,
            compress=True, chunk_size=chunks)
        tq.insert(tasks)
        tq.execute()

    
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--channel', help='Enter channel', required=True)
    parser.add_argument('--suffix', help='Enter suffix to add to the output dir', required=False)
    parser.add_argument('--njobs', help='number of core to use for parallel processing muralus can handle 100 ratto can handle 4', required=False, default=4)
    parser.add_argument('--downsample', help='Enter true or false', required=False, default='true')
    args = parser.parse_args()
    animal = args.animal
    channel = args.channel
    workers = int(args.njobs)
    suffix = args.suffix
    downsample = bool({'true': True, 'false': False}[str(args.downsample).lower()])
    create_downsamples(animal, channel, suffix, downsample,workers)

