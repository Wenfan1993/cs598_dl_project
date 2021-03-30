# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:48:41 2021

@author: Wenxi
"""

from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float
from scipy import ndimage as nd
from skimage import io
from matplotlib import pyplot as plt
import scipy.io as scio
import os

#change the variable path_to_image to your local path
path_to_image = '..\data'
os.chdir(path_to_image)

mat_contents_mask = scio.loadmat('training_data_seg1.mat')
s_xt_label = mat_contents_mask['s_xt_label']
image_obj = s_xt_label[0,:,:]
image_obj.shape
plt.imshow(image_obj)

#guassian_img = nd.gaussian_filter(image_obj, sigma = 5)
#guassian_img.shape
#plt.imshow(guassian_img)
