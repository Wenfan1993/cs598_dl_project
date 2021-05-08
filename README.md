# cs598_dl_project- MRI Denoising & Super Resolutions with GANs 

Our project implements GANs model, in addition to the two base models - BM3D & Guassian Filter, to achieve the removal of additive noise of MRI images.

Please see [Colab page](https://colab.research.google.com/drive/1JQ7rJShfHBGl-DF3VG9ujV62y774dAYg?usp=sharing) for the end-to-end implementation and evaluation workflow

### Please see below the structure of hte implementation:
1. For the GANs model
    a. The data (inputs/targets) are at the directory `cs598_dl_project/project_model/MPRAGE_recon_UNet`.
    b. The Unet implementation details are at `project_model/models/networks.py`, class Unet. (the model implementation is within `pix2pix_model.py`, as consistent with the original pix2pix project)
    c. To train the model, run `project_model/train.py`
    d. To test the model, run `project_model/test.py`
2. The baseline models are implemented at `model_implementation_comparison.py`. The parameter optimization functtions are included at `bm3d_demos/bm3d_param_selection.py` for BM3D model, and `guassian_param_selection.py` for Guanssian Filter model.
3. The evaluation of the baseline and GANs model is included in model_implementation_comparison.py, over 10 test object.

### Acknowledgments
Other than the customization above, the code borrowed from:
1. [Pix2Pix](https://github.com/phillipi/pix2pix), for the GAN architecture reference.
