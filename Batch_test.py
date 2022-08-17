import tifffile as tfi
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage.morphology import binary_opening, binary_erosion
import skimage.morphology as sm
from skimage import measure
import os
from scipy.ndimage.morphology import binary_fill_holes
import utils.display_and_xml as dis
import pandas as pd
import glob

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

    mat_box = []
    offset_val = []
    object_num = []
    file_name = []

    mat_box_list = dis.seq(1,51,2)

    pixels = tfi.imread(args.file)
    pixels_float = pixels.astype('float64')
    ch = pixels_float / 65535.000
    dict = {'test':len(ch)}
    df = pd.DataFrame(dict)
    df.to_csv(args.file +'.csv')

    # # set specific offset range)
    # s = np.random.normal(np.median(ch)/10, np.median(ch)/10, 10000)
    # offset_range = dis.seq(np.min(s),np.max(s),400)
    # # set block box
    # count = 0
    # for offset in offset_range:
    #     ind = dis.unique_rand(1,25,1)
    #     nmask = threshold_local(ch, mat_box_list[ind[0]], "mean", offset)
    #     blank = np.zeros(np.shape(ch))
    #     nmask2 = ch > nmask
    #     #plt.imshow(nmask2,'gray')
    #     if inv:
    #         nmask2 = np.invert(nmask2)
    #         nmask4 = nmask2
    #     else:
    #         nmask3 = binary_opening(nmask2, structure=np.ones((3, 3))).astype(np.float64)
    #         nmask4 = binary_fill_holes(nmask3)
    #     label_objects = sm.label(nmask4, background=0)
    #     info_table = pd.DataFrame(
    #         measure.regionprops_table(
    #             label_objects,
    #             intensity_image=ch,
    #             properties=['area', 'label','coords','centroid'],
    #         )).set_index('label')
    #     #info_table.hist(column='area', bins=100)
    #     # remove small objects - test data frame of small objects
    #     offset_val.append(offset)
    #     object_num.append(len(info_table))
    #     mat_box.append(mat_box_list[ind[0]])
    #     file_name.append(args.file)
    #     count += 1
    # dict = {'image_name':args.file,'object_num':object_num,'block_size':mat_box,'offset':offset_val}
    # df = pd.DataFrame(dict)
    # df.to_csv(args.file +'.csv')