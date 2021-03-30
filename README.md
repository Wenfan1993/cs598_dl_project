# CS598_Project

1. use the below code to visualize the images

from matplotlib import pyplot as plt
import scipy.io as io
import os
#change the variable path_to_image to your local path
path_to_image = r'C:\Users\Wenxi\Desktop\Courses\DL\project\project_repo\CS598_Project\data'
os.chdir(path_to_image)

mat_contents_mask = io.loadmat('training_data_seg1.mat')

s_xt_label = mat_contents_mask['s_xt_label']

plt.imshow(s_xt_label[0,:,:])

