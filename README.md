# AutoSamp
 AutoSamp: Autoencoding MRI Sampling via Variational Information Maximization (https://arxiv.org/abs/2306.02888) 


This repository contains TensorFlow2 implementation of AutoSamp for joint optimization of sampling pattern and reconstruction for MRI. The repository can also used for general DL based Non-Cartesian reconstruction for specified trajectories (more information available at [Training](#Training)).

Currently supports  
- Encoder:  
    - Non-uniform FFT (nuFFT) encoding  
    - Non-uniform DFT matrix encoding
    - Seperable ky-kz encoding as in [J-MoDL](https://github.com/hkaggarwal/J-MoDL)
- Decoder:  
    - Non-Cartesian Unrolled Proximal Gradient Descent
    - U-Net

For nuFFT, we use [tfnufft](https://github.com/alkanc/tfnufft) library that is copied under `tfnufft/`.

# Requirements
This package is tested with
- Python 3.9
- TensorFlow >= 2.5
- CUDA 11.2
- CUDNN 8.1

# Installation
To avoid CUDA related issues cudatoolkit and cudnn can be installed via conda:
 
    conda env create -f environment.yaml
    conda activate autosamp
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
    echo 'export TEMP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$TEMP_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
    echo 'unset TEMP_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
    conda activate autosamp

This will create a new Conda environment named `autosamp`, install the relevant CUDA libraries and configure the environment variables. This step is optional if you already have the CUDA libraries set up.
You can then install the required Python libraries using
<pre><code>pip install -r requirements.txt
</code></pre>

# Data Preparation
Stanford FSE knees data can be downloaded and converted into TFRecords format by running:
<pre><code>python -m autosamp.data_prep.data_prep_stanfordknees
</code></pre>
fastMRI knee and brain data can be processed and converted into TFRecords format by running:
<pre><code>python -m autosamp.data_prep.data_prep_fastmri &ltinput_data_path> &ltoutput_data_path> --test_perc &lttest_perc>
</code></pre>
where `input_data_path` is the top level directory where fastMRI (knee or brain) data was downloaded and `test_perc` defines the percentage of validation dataset to be used as test dataset.

The data_prep scripts are adapted from [dl-cs](https://github.com/MRSRL/dl-cs) repository and require [Berkeley Advanced Reconstruction Toolbox (BART)](https://mrirecon.github.io/bart/) for coil sensitivity estimation.

# Training
To start a training session, run the following script:

    python -m scripts.main --mode=train --flagfile=flagfiles/sample_flagfile_5x.txt

An example flagfile can be found at *sample_flagfile.txt*. The training session stores event files that contain loss curves, reconstructed images and optimized samples for each epoch inside `exp_dir` for visualization using TensorBoard.

The repository can also used for general DL based Non-Cartesian recon by setting `--nooptimize_sampling` flag and specifying a trajectory using `--fixed_mask_coordinates_path`.

Flags are described below:

## File options
`exp_dir`: top directory for checkpoints, events files and results  
`exp_name`: experiment name  
`data_dir`: directory of dataset to be used (in TFRecords format)  
`data_source`: data source including 'knees', 'knees_full', 'fastmri'  

## Training options
`num_epochs`: number of training epochs  
`batch_size`: number of datapoints per batch  
`optimizer`: 'sgd', 'adam', 'adamax'  
`lr`: learning rate for optimizer  
`lr_enc`: learning rate for optimizer for the encoder. If None, uses the optimizer specified with the learning rate `--lr` is used for both encoder and decoder  
`loss`: loss function to use: mse (l2) for Gaussian observation model, mae (l1) for Laplacian observation model  

## Model options
`undersamp_ratio`: undersampling ratio  
`noise_std`: std. of noise  
`enc_arch`: encoder module to use, can be one of tfnufft/matrix/matrix1d/matrix/matrix_jmodl_2d/fft  
`coords_initializer`: initializer for k-space coordinates: supports uniform/circular_uniform/small_circular_uniform/normal/mask/mask_coordinates  
`coords_calib_size`: size of calibration region for k-space coordinates  
`fixed_mask_path`: path for the sampling mask for coordinate initializion, overwrites undersamp_ratio and coords_initializer  
`fixed_mask_coordinates_path`: path for the sampling mask coordinates for coordinate initializion, overwrites coords_initializer. Trajectory coordinates must be in [-H/2, H/2] for ky and [-W/2, W/2] for kx, where [H,W] is the image size  
`optimize_sampling`: (bool) optimize k-space sampling coordinates  
`tfnufft_oversamp`: oversampling factor for tfnufft operations  
`tfnufft_width`: interpolation kernel full-width in terms of oversampled grid for tfnufft operations  
`dec_arch`: decoder module to use, can be one of unrolled/unet  
`unrolled_steps`: number of grad steps for unrolled algorithms  
`unrolled_num_resblocks`: number of ResBlocks per iteration  
`unrolled_num_filters`: number of features for the convolutional layers in each ResBlock  
`unrolled_kernel_size`: kernel size for the convolutional layers in each ResBlock  
`unrolled_act_fun`: activation function to use in unrolled network relu/lrelu  
`unrolled_normalization_layer`: normalization layer to use: instance/layer/batch/None  
`unrolled_share_proxresblocks`: (bool) share the ProxResBlocks across iterations  

# About
If you use AutoSamp for your work, please consider citing the following work:

    @article{alkan2024autosamp, 
             title={AutoSamp: Autoencoding k-space Sampling via Variational Information Maximization for 3D MRI},
             author={Alkan, Cagan and Mardani, Morteza and Liao, Congyu and Li, Zhitao and Vasanawala, Shreyas S and Pauly, John M},
             journal={IEEE Transactions on Medical Imaging},
             year={2024},
             publisher={IEEE}
    }
