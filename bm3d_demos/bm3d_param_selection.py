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

def bm3d_param_selection():
    outputs = sio.loadmat(r'./project_model/results/test/test_latest.mat')
    
    sigmaX = np.exp(np.arange(-10,10))
    
    base_psnrs = []
    
    for x in sigmaX:
      for  i in range(len(outputs)):
        orig_image = outputs['real_B'][i]
        noisy_img = outputs['real_A'][i]
      
        #test base model
        y_pred = bm3d(noisy_img, x)
      
        psnr = get_psnr(orig_image, y_pred)
        base_psnrs.append({'param':x,
                                'rate':psnr})
    
    best_param = sorted(base_psnrs, key = lambda x: x['rate'], reverse=True)[0]

    return best_param