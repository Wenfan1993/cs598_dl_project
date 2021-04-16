# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:26:38 2021

@author: Wenxi
"""

import numpy as np
from bm3d import bm3d
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
os.chdir(r'./bm3d_demos')
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
os.chdir(r'..')
os.getcwd()

sigma = 0.001
i=0
noisy_img = np.load(f'./project_model/MPRAGE_recon_UNet/training_data_input/training_data_seg{i}.npy')
noisy_img = noisy_img[0,:,:]
y_pred= bm3d(noisy_img, sigma)
orig_image = np.load(f'./project_model/MPRAGE_recon_UNet/training_data_label/training_data_seg{i}.npy')
orig_image = orig_image[0,:,:]

psnr = get_psnr(orig_image, y_pred)
print("PSNR:", psnr)

mae = mean_absolute_error(orig_image, y_pred)
print("mae:", mae)

mse = mean_squared_error(orig_image, y_pred)
print("mae:", mae)
