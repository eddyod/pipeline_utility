import sys
import argparse
from datetime import time

from utilities.data_manager_v2 import DataManager
from utilities.utilities2015 import load_ini, create_parent_dir_if_not_exists, execute_command

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Linearly normalize intensity to between 0 and 255')

parser.add_argument("input_spec", type=str, help="Input specification")
parser.add_argument("out_version", type=str, help="Output image version")
args = parser.parse_args()


out_version = args.out_version

input_spec = load_ini(args.input_spec)
stack = input_spec['stack']
image_name_list = input_spec['image_name_list']
if image_name_list == 'all':
    image_name_list = DataManager.load_sorted_filenames(stack=stack)[0].keys()
prep_id = input_spec['prep_id']
if prep_id == 'None':
    prep_id = None
resol = input_spec['resol']
version = input_spec['version']
if version == 'None':
    version = None

for img_name in image_name_list:
    t = time.time()

    in_fp = DataManager.get_image_filepath(stack=stack, resol=resol, version=version, fn=img_name)
    out_fp = DataManager.get_image_filepath(stack=stack, resol=resol, version=out_version, fn=img_name)
    create_parent_dir_if_not_exists(out_fp)
        
    cmd = """convert "%(in_fp)s" -normalize -depth 8 "%(out_fp)s" """ % {'in_fp': in_fp, 'out_fp': out_fp}
    execute_command(cmd)
    
    sys.stderr.write("Intensity normalize: %.2f seconds.\n" % (time.time() - t))
