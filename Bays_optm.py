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

os.chdir('/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash/utils')
import utils.display_and_xml as dis

'''
Bayesian modules
'''

from scipy import stats
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt



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

#lets aggregate all the data:
path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash/Bayesian_optemisation/output/box_grad/'
tables = glob.glob(os.path.join(path,'*.csv'))
# marge all the tables together

class rbind:
    def __init__(self,csv_file,path):
        self.csv_file = csv_file
        self.path = path
    def load_table(self):
        os.chdir(self.path)
        if len(self.csv_file) > 1:
            newDF = pd.DataFrame()
            for csv in self.csv_file:
                temp = pd.read_csv(csv)
                newDF = pd.concat([newDF, temp])
            return newDF
        else:
            temp = pd.read_csv(self.csv_file)
            return temp
table_gen = rbind(csv_file=tables, path=path)
df = table_gen.load_table()
#colnames
print(df.columns.tolist())

# scatter plot:
sns.scatterplot(data=df, x="offset", y="object_num")
sns.scatterplot(data=df, x="block_size", y="object_num")
#
#Define the Gaussian function
def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y
mean_x = np.mean(df['offset'])
#0.00025524309639343007
sd_x = np.std(df['offset'])
#0.010005800140397514

#
#
# df.head(1)
# plt.hist(df['object_num'], bins = 'fd')
# plt.xlabel('object_num')
# plt.ylabel('Count')
# plt.title('Distribution of Final object_num')
#
#
# plt.hist(df['offset'], bins = 'fd')
# plt.xlabel('offset')
# plt.ylabel('Count')
# plt.title('Distribution of Final  offset')
#
# un_image_name  = df['image_name'].unique()
# len(un_image_name)
#
# sns.kdeplot(df.loc[df['image_name'] == un_image_name[0],'object_num'],label = un_image_name[0], shade = True)
# sns.kdeplot(df.loc[df['image_name'] == un_image_name[1],'object_num'],label = un_image_name[1], shade = True)
# sns.kdeplot(df.loc[df['image_name'] == un_image_name[2],'object_num'],label = un_image_name[2], shade = True)
# sns.kdeplot(df.loc[df['image_name'] == un_image_name[3],'object_num'],label = un_image_name[3], shade = True)
# sns.kdeplot(df.loc[df['image_name'] == un_image_name[4],'object_num'],label = un_image_name[4], shade = True)
# sns.kdeplot(df.loc[df['image_name'] == un_image_name[5],'object_num'],label = un_image_name[5], shade = True)
# sns.kdeplot(df.loc[df['image_name'] == un_image_name[6],'object_num'],label = un_image_name[6], shade = True)
# plt.xlabel('object_num')
# plt.ylabel('Density')
# plt.title('Density Plot of Final Grades by Image')
#
#
# # corrlation plots
# #https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
# sns.pairplot(df)

''':parameter
what are the relis=hensheep between
'''



from sklearn.model_selection import train_test_split
labels = df.loc[:,['object_num',]]
df_feature = df.loc[:,['object_num','block_size','offset']]
X_train, X_test, y_train, y_test = train_test_split(df_feature, labels,
                                                   test_size = 0.25,
                                                    random_state=42)
X_train.head()

#object_num_median = np.median(df.loc[:,'object_num'])

# test if the feature effect the median
# X_train is our training data, we will make a copy for plotting
# X_plot = X_train.copy().to_frame()

# Compare grades to the median
# X_plot['relation_median'] = (X_plot.loc[:,'object_num'] >= object_num_median)
# X_plot['object_num'] = X_plot['object_num'].replace({True: 'above',
#                                            False: 'below'})
# # Plot all variables in a loop
# plt.figure(figsize=(12, 12))
# for i, col in enumerate(X_plot.columns[:-1]):
#     plt.subplot(3, 2, i + 1)
#     subset_above = X_plot[X_plot['relation_median'] == 'above']
#     subset_below = X_plot[X_plot['relation_median'] == 'below']
#     sns.kdeplot(subset_above[col], label='Above Median')
#     sns.kdeplot(subset_below[col], label='Below Median')
#     plt.legend()
#     plt.title('Distribution of %s' % col)

plt.tight_layout()

plt.figure(figsize=(12, 12))

formula = 'object_num ~ offset + block_size'

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)
#
# with pm.Model() as normal_model:
#     # The prior for the data likelihood is a Normal Distribution
#     family = pm.glm.families.Normal()
#
#     # Creating the model requires a formula and data (and optionally a family)
#     pm.GLM.from_formula(formula, data=X_train, family=family)
#
#     # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
#     normal_trace = pm.sample(draws=100, chains=2, tune=5)

from pymc3.glm import GLM




with pm.Model():
    GLM.from_formula('object_num ~ block_size + offset',df_feature)
    trace = pm.sample()

#
# confirmed = df.loc[:,'object_num'].values
# with pm.Model() as model_expl:
#     offset = pm.Normal('offset',mu = 0.002, sigma=0.001)
#     block_size = pm.Binomial('block_size', 51, 15)
#     object_num = offset + block_size
#     eps = pm.HalfNormal('eps',0.5)
#
#     pm.Normal('obs',mu = object_num,sigma=eps,
#               observed=confirmed)
#
# with model_expl:
#     prior_pred = pm.sample_prior_predictive()
#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(prior_pred['obs'].T, color='0.5',alpha=0.1)
#
# plt.plot(prior_pred['obs'].T, color='0.5',alpha=0.1)
