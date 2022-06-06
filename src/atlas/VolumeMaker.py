"""
This gets the CSV data of the 3 foundation brains' annotations.
These annotations were done by Lauren, Beth, Yuncong and Harvey
(i'm not positive about this)
The annotations are full scale vertices.

This code takes the contours and does the following:
1. create 3d volumn from the contours
2. find the center of mass
3. saving COMs in the database, saving COM and volumns in the file system
"""
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage.measurements import center_of_mass
from atlas.BrainStructureManager import BrainStructureManager
from scipy.interpolate import splprep, splev

class VolumeMaker(BrainStructureManager):
    def __init__(self,animal, *args, **kwargs):
        BrainStructureManager.__init__(self,animal = animal, *args, **kwargs)
        self.load_aligned_contours()
    
    def get_contours_dimension(self,contours):
        section_mins = []
        section_maxs = []
        for _, contour_points in contours.items():
            contour_points = np.array(contour_points)
            section_mins.append(np.min(contour_points, axis=0))
            section_maxs.append(np.max(contour_points, axis=0))
        min_z = min([int(i) for i in contours.keys()])
        min_x,min_y = np.min(section_mins, axis=0)
        max_x,max_y = np.max(section_maxs, axis=0)
        xspan = max_x - min_x
        yspan = max_y - min_y
        PADDED_SIZE = (int(yspan+5), int(xspan+5))
        return min_x,min_y,min_z,PADDED_SIZE

    def calculate_origin_COM_and_volume(self,contour_for_structurei,structurei):
        contour_for_structurei = self.sort_contours(contour_for_structurei)
        min_x,min_y,min_z,PADDED_SIZE = self.get_contours_dimension(contour_for_structurei)
        volume = []
        for _, contour_points in contour_for_structurei.items():
            vertices = np.array(contour_points) - np.array((min_x, min_y))
            contour_points = (vertices).astype(np.int32)
            volume_slice = np.zeros(PADDED_SIZE, dtype=np.uint8)
            volume_slice = cv2.polylines(volume_slice, [contour_points], isClosed=True, color=1, thickness=1)
            volume_slice = cv2.fillPoly(volume_slice, pts=[contour_points], color=1)
            volume.append(volume_slice)
        volume = np.array(volume).astype(np.bool8)
        volume = np.swapaxes(volume,0,2)
        com = np.array(center_of_mass(volume))
        self.COM[structurei] = (com+np.array((min_x,min_y,min_z)))*self.pixel_to_um
        self.origins[structurei] = np.array((min_x,min_y,min_z))
        self.volumes[structurei] = volume

    def compute_COMs_origins_and_volumes(self):
        self.check_attributes(['structures'])
        for structurei in tqdm(self.structures):
            contours_of_structurei = self.aligned_contours[structurei]
            self.calculate_origin_COM_and_volume(contours_of_structurei,structurei)
        
    def show_steps(self):
        self.plot_volume_stack()
    
    def sort_contours(self,contour_for_structurei):
        sections = [int(section) for section in contour_for_structurei]
        section_order = np.argsort(sections)
        keys = np.array(list(contour_for_structurei.keys()))[section_order]
        values = np.array(list(contour_for_structurei.values()), dtype=object)[section_order]
        return dict(zip(keys,values))

    def bspline_interpolate_contour(self,points):
        tck, u = splprep(points, u=None, s=2, per=2)
        u_new = np.linspace(u.min(), u.max(), 50)
        x_array, y_array = splev(u_new, tck, der=0)
        arr_2d = np.concatenate([x_array[:, None], y_array[:, None]], axis=1)
        return arr_2d


    def eliminate_duplicates(self,point_list):
        if point_list.shape[0] == 2:
            point_list = point_list.T
        point_list = [list(i) for i in point_list]
        output = []
        for _ in range(len(point_list)):
            elementi = point_list.pop(0)
            if not elementi in output:
                output.append(elementi)
            else:
                while elementi in output:
                    elementi = [i+0.01 for i in elementi]
                output.append(elementi)
        output = np.array(output)
        return output

if __name__ == '__main__':
    animals = ['MD589','MD585', 'MD594']
    for animal in animals:
        volumemaker = VolumeMaker(animal)
        volumemaker.load_aligned_contours()
        volumemaker.compute_COMs_origins_and_volumes()
        # volumemaker.show_steps()
        volumemaker.save_coms()
        volumemaker.save_origins()
        volumemaker.save_volumes()