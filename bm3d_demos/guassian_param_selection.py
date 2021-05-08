# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:21:31 2021

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
import cv2
from skimage import io, img_as_float
from skimage.filters import gaussian

def guassian_param_selection():
    outputs = sio.loadmat(r'./project_model/results/test/test_latest.mat')
    
    sigmaX = np.exp(np.arange(-10,10))
    sigmaY = np.exp(np.arange(-10,10))
    kernal  = [i for i in  range(1,12) if i%2==1]
    
    guassian_psnrs = []
    
    for x in sigmaX:
      for y in sigmaY:
        for k in kernal:
          for  i in range(len(outputs)):
            orig_image = outputs['real_B'][i]
            noisy_img = outputs['real_A'][i]
          
            #test guassian model
            y_pred_guassian = cv2.GaussianBlur(noisy_img,
                                               ksize = (k,k),
                                               sigmaX = x,
                                               sigmaY = y,
                                               borderType = cv2.BORDER_DEFAULT)
          
            psnr = get_psnr(orig_image, y_pred_guassian)
            guassian_psnrs.append({'param':(x,y,k),
                                   'rate':psnr})
                    
    best_param = sorted(guassian_psnrs, key = lambda x: x['rate'], reverse=True)[0]

    return best_param