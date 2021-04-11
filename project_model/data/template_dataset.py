"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from scipy.io import loadmat
import h5py


class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     """Add new dataset-specific options, and rewrite default values for existing options.

    #     Parameters:
    #         parser          -- original option parser
    #         is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    #     Returns:
    #         the modified parser.
    #     """
    #     parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
    #     parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
    #     return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)

        self.high = np.zeros((256, 256, 3580*2))
        self.low = np.zeros((256, 256, 3580*2))
        high_mat = loadmat(opt.dataroot + '/T2_train_0920_p1.mat')
        self.high = high_mat['t2_wm']
        low_mat = loadmat(opt.dataroot + '/T2_train_0920_p1.mat')
        self.low = low_mat['mp_wm']
        print(self.high.shape)
        print(self.low.shape)
        
        if opt.center == True:
            if opt.phase == 'train':
                self.high = self.high[:, :, 0:3500*2]
                self.low = self.low[:, :, 0:3500*2]
            else:
                self.high = self.high[:, :, 3500*2:]
                self.low = self.low[:, :, 3500*2:]
        else:
            if opt.phase == 'train':
                self.high = self.high[:, :, 0:3500*2]
                self.low = self.low[:, :, 0:3500*2]
            else:
                self.high = self.high[:, :, 3500*2:]
                self.low = self.low[:, :, 3500*2:]

        print(self.high.shape)
        print(self.low.shape)

        self.length = self.high.shape[2]

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

#
        print("Number of data: ", self.length)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = str(index)    # needs to be a string

        A = self.low[:, :, index]
        B = self.high[:, :, index]

        A = A.reshape((1, 256, 256))
        B = B.reshape((1, 256, 256))

        data_A = torch.from_numpy(A).float()
        data_B = torch.from_numpy(B).float()

        return {'A': data_A, 'B': data_B, 'A_paths': path, 'index': index}

    def __len__(self):
        """Return the total number of images."""
        return self.length
