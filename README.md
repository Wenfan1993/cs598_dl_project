# cs598_dl_project

Our project implements GANs model, in addition to the two base models - BM3D & Guassian Filter, to achieve the removal of additive noise of MRI images.

Please see colab page https://colab.research.google.com/drive/1JQ7rJShfHBGl-DF3VG9ujV62y774dAYg?usp=sharing for the end-to-end implementation and evaluation workflow

Please see below the structure of hte implementation:
1. For the GANs model
    a. The data (inputs/targets) are at the directory cs598_dl_project/project_model/MPRAGE_recon_UNet.
    b. The Unet implementation details are at project_model/models/networks.py, class Unet
    

Acknowledgments
Other than the customization above, the code borrowed from:
1. https://github.com/phillipi/pix2pix, for the GAN architecture reference;
2. 
