import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        # self.new_im_A = Image.new('L', (240, 128 * 200))
        # self.new_im_B = Image.new('L', (240, 128 * 200)) 
        # A_offset = 0
        # B_offset = 0
        # for path in self.AB_paths: 
        #     file_name = os.path.basename(path)
        #     num = int(os.path.splitext(file_name)[0])
        #     AB = Image.open(path)
        #     w, h = AB.size
        #     w2 = int(w / 2)
        #     A = AB.crop((0, 0, w2, h))
        #     B = AB.crop((w2, 0, w, h))
        #     self.new_im_A.paste(A, (0, A_offset))
        #     self.new_im_B.paste(B, (0, B_offset))
        #     A_offset += h
        #     B_offset += h

        # self.new_im_A.save('./test_A.png')
        # self.new_im_B.save('./test_B.png')

        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # self.A_trans = A_transform(self.new_im_A)
        # self.B_trans = B_transform(self.new_im_B)
        # print(self.A_trans.size())
        # print(self.B_trans.size())

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path)

        # print("AB size: ", AB.size)

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # A = self.A_trans[0:240, index*128:(index+1)*128]
        # B = self.B_trans[0:240, index*128:(index+1)*128]

        # print(A.size())
        # print(B.size())
        # AB_path = self.AB_paths[index]

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)