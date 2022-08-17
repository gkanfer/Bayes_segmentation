import tifffile as tfi
import skimage.measure as sme
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
# from PIL import fromarray
from numpy import asarray
from skimage import data, io
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_opening, binary_erosion
from skimage.morphology import disk, remove_small_objects
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank, gaussian, sobel
from skimage.util import img_as_ubyte
from skimage import data, util
from skimage.measure import regionprops_table
from skimage.measure import perimeter
from skimage import measure
from skimage.exposure import rescale_intensity, histogram
from skimage.feature import peak_local_max
import os
import glob
import pandas as pd
from pandas import DataFrame
from scipy.ndimage.morphology import binary_fill_holes
from skimage.viewer import ImageViewer
from skimage import img_as_float
import time
import base64
from datetime import datetime

import utils.display_and_xml as dis

'''
Bayesian modules
'''

from scipy import stats
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import seaborn as sns
import pandas as pd
from theano import shared
from sklearn import preprocessing



def segmentation(image,mat,offset_inp):
    nmask = threshold_local(ch, mat, "mean", offset_inp)
    blank = np.zeros(np.shape(ch))
    nmask2 = ch > nmask
    nmask3 = binary_opening(nmask2, structure=np.ones((3, 3))).astype(np.float64)
    nmask4 = binary_fill_holes(nmask3)
    label_objects = sm.label(nmask4, background=0)
    info_table = pd.DataFrame(
        measure.regionprops_table(
            label_objects,
            intensity_image=ch,
            properties=['area', 'label', 'coords', 'centroid'],
        )).set_index('label')
    return info_table


'''
Part one: 
https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-1-7d0ad817fca5
Part two:
https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e
----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------
Bayesian optimisation for segmentation parameters
offset optimisation
1) Initiate value - image intensity mean divided by 10
2) Calculate number of object detected per offset value
'''


if (__name__ == "__main__"):
    import argparse
    parser = argparse.ArgumentParser(description='AIPS activation')
    parser.add_argument('--file', dest='file', type=str, required=True,
                        help="The name of the image to analyze")
    args = parser.parse_args()
    pixels = tfi.imread(os.path.join(args.path, args.file))
    pixels_float = pixels.astype('float64')
    ch = pixels_float / 65535.000

    mat_box = []
    offset_val = []
    object_num = []
    file_name = []
    count = 2000
    #find intial value
    s = np.random.normal(np.median(ch) / 100, np.median(ch) / 1000, 10000)
    offset_range = dis.seq(np.min(s), np.max(s), 1000)
    mat_box_list = dis.seq(1, 99, 2)
    while count < 2000:
        for i in range(50):
            ind_offset = dis.unique_rand(1, np.shape(offset_range)[0], 1)
            test_table = segmentation(image=ch, mat=mat_box_list[i], offset_inp=offset_range[ind_offset[0]])





    mat_box_list = dis.seq(1,51,2)


    # set specific offset range)


    s = np.random.normal(np.median(ch)/10, np.median(ch)/10, 10000)
    offset_range = dis.seq(np.min(s),np.max(s),400)
    # set block box
    count = 0
    for offset in offset_range:
        ind = dis.unique_rand(1,25,1)
        nmask = threshold_local(ch, mat_box_list[ind[0]], "mean", offset)
        blank = np.zeros(np.shape(ch))
        nmask2 = ch > nmask
        #plt.imshow(nmask2,'gray')
        if inv:
            nmask2 = np.invert(nmask2)
            nmask4 = nmask2
        else:
            nmask3 = binary_opening(nmask2, structure=np.ones((3, 3))).astype(np.float64)
            nmask4 = binary_fill_holes(nmask3)
        label_objects = sm.label(nmask4, background=0)
        info_table = pd.DataFrame(
            measure.regionprops_table(
                label_objects,
                intensity_image=ch,
                properties=['area', 'label','coords','centroid'],
            )).set_index('label')
        #info_table.hist(column='area', bins=100)
        # remove small objects - test data frame of small objects
        offset_val.append(offset)
        object_num.append(len(info_table))
        mat_box.append(mat_box_list[ind[0]])
        file_name.append(args.file)
        count += 1
    dict = {'image_name':args.file,'object_num':object_num,'block_size':mat_box,'offset':offset_val}
    df = pd.DataFrame(dict)
    df.to_csv(args.file +'.csv')