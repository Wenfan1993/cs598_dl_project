# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:26:38 2021

@author: Wenxi
"""
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from bm3d import bm3d
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
os.chdir(r'./bm3d_demos')
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
os.chdir(r'..')
os.getcwd()


num_test_images = 10

# base model parameter
sigma = 0.001
#dl model outputs
outputs = sio.loadmat(r'./project_model/results/test/test_latest.mat')


base_psnrs = []
base_maes = []
base_mses = []
dl_psnrs = []
dl_maes = []
dl_mses = []

for i in range(num_test_images):
    orig_image = outputs['real_B'][i]
    noisy_img = outputs['real_A'][i]
    
    #test base model
    y_pred= bm3d(noisy_img, sigma)

    psnr = get_psnr(orig_image, y_pred)
    base_psnrs.append(psnr)
    
    mae = mean_absolute_error(orig_image, y_pred)
    base_maes.append(mae)
    
    mse = mean_squared_error(orig_image, y_pred)
    base_mses.append(mse)
    
    #test dl model
    y_pred_dl =  outputs['fake_B'][i]

    psnr_dl = get_psnr(orig_image, y_pred_dl)
    dl_psnrs.append(psnr_dl)
    
    mae_dl = mean_absolute_error(orig_image, y_pred_dl)
    dl_maes.append(mae_dl)
    
    mse_dl = mean_squared_error(orig_image, y_pred_dl)
    dl_mses.append(mse_dl)
    
    
base_psnr = np.array(base_psnrs).mean()
base_mae = np.array(base_maes).mean()
base_mse = np.array(base_mses).mean()

dl_psnr = np.array(dl_psnrs).mean()
dl_mae = np.array(dl_maes).mean()
dl_mse = np.array(dl_mses).mean()

print(f'Evaluation Results on {num_test_images} images')
print(f'\n******\nThe base line model evaluation results are:\n1.PSNR: {base_psnr}\n2.MAE: {base_mae}\n3.MSE: {base_mse}')
print(f'\n******\nThe DL model evaluation results are:\n1.PSNR: {dl_psnr}\n2.MAE: {dl_mae}\n3.MSE: {dl_mse}')
