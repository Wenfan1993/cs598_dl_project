# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:48:41 2021

@author: Wenxi
"""

from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float
from scipy import ndimage as nd

from matplotlib import pyplot as plt
import scipy.io as io
import os

#change the variable path_to_image to your local path
path_to_image = '.\data'
os.chdir(path_to_image)

mat_contents_mask = io.loadmat('training_data_seg1.mat')

s_xt_label = mat_contents_mask['s_xt_label']

plt.imshow(s_xt_label[0,:,:])

