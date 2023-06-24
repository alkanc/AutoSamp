from absl import app
from absl import flags

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import autosamp.mrutils as mrutils
import autosamp.subsample as subsample
from tfnufft.fourier import nufft, nufft_adjoint

FLAGS = flags.FLAGS

class MR_UAE(tf.keras.Model):

    def __init__(self, config=None):
        super(MR_UAE, self).__init__()
        if config is None:
            config = FLAGS
        
        if len(config.undersamp_ratio)==1 and config.enc_arch!='fft':
            self.undersamp_ratio = config.undersamp_ratio[0]
        else:
            self.undersamp_ratio = config.undersamp_ratio
        self.simulate_from_kspace = config.simulate_from_kspace if hasattr(config, 'simulate_from_kspace') else False
        self.noise_std = config.noise_std
        self.enc_arch = config.enc_arch
        self.dec_arch = config.dec_arch
        self.coords_initializer = config.coords_initializer
        self.coords_calib_size = config.coords_calib_size
        self.fixed_mask_path = config.fixed_mask_path
        self.fixed_mask_coordinates_path = config.fixed_mask_coordinates_path
        self.optimize_sampling = config.optimize_sampling
        self.use_dcomp = config.use_dcomp
        self.fft_mask_type = config.fft_mask_type
        self.fft_mask_train_seed = config.fft_mask_train_seed
        self.fft_mask_val_seed = config.fft_mask_val_seed
        self.fft_mask_center_fractions = config.fft_mask_center_fractions
        self.fft_mask_calib_sizes = config.fft_mask_calib_sizes
        self.tfnufft_oversamp = config.tfnufft_oversamp
        self.tfnufft_width = config.tfnufft_width

        # Encoding
        if self.enc_arch == 'tfnufft':
            if self.simulate_from_kspace:
                self.encode = self.encode_tfnufft_from_kspace
            else:
                self.encode = self.encode_tfnufft
            self.forward = self.tfnufft_forward
            self.adjoint = self.tfnufft_transpose
            
        elif self.enc_arch == 'matrix':
            self.encode = self.encode_matrix_nudft
            self.forward = self.matrix_nudft_forward
            self.adjoint = self.matrix_nudft_transpose

        elif self.enc_arch == 'matrix_jmodl_2d':
            if self.simulate_from_kspace:
                self.encode = self.encode_matrix_jmodl_2d_from_kspace
            else:
                self.encode = self.encode_matrix_jmodl_2d
            self.forward = self.matrix_jmodl_2d_forward
            self.adjoint = self.matrix_jmodl_2d_transpose

        elif self.enc_arch == 'matrix1d':
            self.encode = self.encode_matrix_nudft1d
            self.forward = self.matrix_nudft1d_forward
            self.adjoint = self.matrix_nudft1d_transpose
        
        elif self.enc_arch == 'fft':
            self.encode = self.encode_fft
            self.forward = self.fft_forward
            self.adjoint = self.fft_transpose

        else:
            raise ValueError(f'Encoding architecture {self.enc_arch} not supported.')

        # Decoding
        if self.dec_arch == 'unrolled':
            self.recon_fn = self.unrolled_prox
            self.unrolled_steps = config.unrolled_steps
            self.unrolled_num_resblocks = config.unrolled_num_resblocks
            self.unrolled_num_filters = config.unrolled_num_filters
            self.unrolled_kernel_size = config.unrolled_kernel_size
            self.unrolled_normalization_layer = config.unrolled_normalization_layer
            self.unrolled_act_fun = config.unrolled_act_fun
            self.unrolled_share_proxresblocks = config.unrolled_share_proxresblocks

        elif self.dec_arch == 'unet':
            self.recon_fn = self.unet_recon
            self.unet_num_features = config.unet_num_features
            self.unet_num_levels = config.unet_num_levels
            self.unet_kernel_size = config.unet_kernel_size
            self.unet_normalization_layer = config.unet_normalization_layer
            self.unet_act_fun = config.unet_act_fun
        
        else:
            raise ValueError(f'Decoding architecture {self.dec_arch} not supported.')


    def build(self, input_shape):
        "Builds MR_UAE model. Assumes the inputs are NCHW"
        
        img_shape, sensemap_shape = map(lambda x: x.as_list(), input_shape)
        self.h = img_shape[-2]
        self.w = img_shape[-1]
        self.c = sensemap_shape[-3]
        
        # Encoding
        # 3dft case (optimization on ky-kz plane)
        if self.enc_arch == 'matrix' or self.enc_arch == 'tfnufft':
            self.z_dim = int(img_shape[-2]*img_shape[-1]*self.undersamp_ratio)
            self.z_range_h = (-img_shape[-2]/2, img_shape[-2]/2) # (min, max) tuple
            self.z_range_w = (-img_shape[-1]/2, img_shape[-1]/2) # (min, max) tuple

            # initializers
            if self.coords_initializer == 'uniform':
                if self.coords_calib_size is None:
                    self.coords_calib_size = 0
                target_num_samples = self.z_dim - self.coords_calib_size ** 2
                phi_h = np.random.uniform(low=self.z_range_h[0], high=self.z_range_h[1], size=[target_num_samples])
                phi_w = np.random.uniform(low=self.z_range_w[0], high=self.z_range_w[1], size=[target_num_samples])
                # add calib region
                calib_h, calib_w = np.meshgrid(np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               indexing='ij')
                phi_h = np.concatenate([phi_h, np.reshape(calib_h, [-1])], axis=0)
                phi_w = np.concatenate([phi_w, np.reshape(calib_w, [-1])], axis=0)
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)
            elif self.coords_initializer == 'normal': # variable density sampling
                # with truncated normal values more than
                # 2stddevs are discarded and resampled
                if self.coords_calib_size is None:
                    self.coords_calib_size = 0
                target_num_samples = self.z_dim - self.coords_calib_size ** 2
                phi_h = tf.random.truncated_normal(shape=[target_num_samples], mean=0.0, stddev=self.z_range_h[1]/2)
                phi_w = tf.random.truncated_normal(shape=[target_num_samples], mean=0.0, stddev=self.z_range_w[1]/2)
                # add calib region
                calib_h, calib_w = np.meshgrid(np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               indexing='ij')
                phi_h = np.concatenate([phi_h.numpy(), np.reshape(calib_h, [-1])], axis=0)
                phi_w = np.concatenate([phi_w.numpy(), np.reshape(calib_w, [-1])], axis=0)
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)     
            elif self.coords_initializer == 'circular_uniform':
                if self.coords_calib_size is None:
                    self.coords_calib_size = 0
                target_num_samples = self.z_dim - self.coords_calib_size ** 2
                theta = 2*np.pi*np.random.rand(target_num_samples)
                r = np.random.rand(target_num_samples)
                phi_h = np.sqrt(r)*np.sin(theta)*self.z_range_h[1]
                phi_w = np.sqrt(r)*np.cos(theta)*self.z_range_w[1]
                # add calib region
                calib_h, calib_w = np.meshgrid(np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               indexing='ij')
                phi_h = np.concatenate([phi_h, np.reshape(calib_h, [-1])], axis=0)
                phi_w = np.concatenate([phi_w, np.reshape(calib_w, [-1])], axis=0)
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)
            elif self.coords_initializer == 'small_circular_uniform':
                if self.coords_calib_size is None:
                    self.coords_calib_size = 0
                target_num_samples = self.z_dim - self.coords_calib_size ** 2
                theta = 2*np.pi*np.random.rand(target_num_samples)
                r = np.random.rand(target_num_samples) * 0.2
                phi_h = np.sqrt(r)*np.sin(theta)*self.z_range_h[1]
                phi_w = np.sqrt(r)*np.cos(theta)*self.z_range_w[1]
                # add calib region
                calib_h, calib_w = np.meshgrid(np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               indexing='ij')
                phi_h = np.concatenate([phi_h, np.reshape(calib_h, [-1])], axis=0)
                phi_w = np.concatenate([phi_w, np.reshape(calib_w, [-1])], axis=0)
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)
            elif self.coords_initializer == 'smaller_circular_uniform':
                if self.coords_calib_size is None:
                    self.coords_calib_size = 0
                target_num_samples = self.z_dim - self.coords_calib_size ** 2
                theta = 2*np.pi*np.random.rand(target_num_samples)
                r = np.random.rand(target_num_samples) * 0.05
                phi_h = np.sqrt(r)*np.sin(theta)*self.z_range_h[1]
                phi_w = np.sqrt(r)*np.cos(theta)*self.z_range_w[1]
                # add calib region
                calib_h, calib_w = np.meshgrid(np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32), 
                                               indexing='ij')
                phi_h = np.concatenate([phi_h, np.reshape(calib_h, [-1])], axis=0)
                phi_w = np.concatenate([phi_w, np.reshape(calib_w, [-1])], axis=0)
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)
            elif self.coords_initializer == 'half_circle_border':
                theta = 2*np.pi*np.arange(self.z_dim)/self.z_dim
                phi_h = 0.5*np.sin(theta)*self.z_range_h[1]
                phi_w = 0.5*np.cos(theta)*self.z_range_w[1]
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)
            elif self.coords_initializer == 'mask':
                sampling_mask = np.load(self.fixed_mask_path)
                if sampling_mask.shape != (self.h, self.w):
                    raise ValueError('Mask shape and input shape are inconsistent.')
                phi_h_indices, phi_w_indices = np.nonzero(sampling_mask)
                phi_h = phi_h_indices - sampling_mask.shape[0]//2
                phi_w = phi_w_indices - sampling_mask.shape[1]//2
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)
            elif self.coords_initializer == 'mask_coordinates':
                sampling_mask = np.load(self.fixed_mask_coordinates_path)
                if (np.max(sampling_mask[0])>self.w/2 or np.min(sampling_mask[0])<-self.w/2 
                    or np.max(sampling_mask[1])>self.h/2 or np.min(sampling_mask[1])<-self.h/2):
                    raise ValueError('Mask coordinates entries must match the image size.')
                if np.abs(sampling_mask.shape[1]-self.z_dim)/(self.z_dim)>0.01:
                    raise ValueError('Size of mask coordinates does not match undersampling ratio.')
                self.z_dim = sampling_mask.shape[1]
                phi_w = sampling_mask[0]
                phi_h = sampling_mask[1]
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)

            self.phi_h = self.add_weight(name='phi_h',
                                        shape=[self.z_dim],
                                        dtype=tf.float32,
                                        initializer=var_init_h,
                                        trainable=self.optimize_sampling,
                                        constraint=ModuloConstraint(self.z_range_h[1]*2))
            self.phi_w = self.add_weight(name='phi_w',
                                        shape=[self.z_dim],
                                        dtype=tf.float32,
                                        initializer=var_init_w,
                                        trainable=self.optimize_sampling,
                                        constraint=ModuloConstraint(self.z_range_w[1]*2))
            
            if self.use_dcomp:
                raise ValueError('dcomp is not supported.')
        
        # 3dft case with jmodl type of optimization (optimization on ky-kz plane)
        elif self.enc_arch == 'matrix_jmodl_2d':

            self.z_dim_h = int(img_shape[-2]*np.sqrt(self.undersamp_ratio))
            self.z_dim_w = int(int(img_shape[-2]*img_shape[-1]*self.undersamp_ratio) / self.z_dim_h)
            self.z_dim = self.z_dim_h * self.z_dim_w

            self.z_range_h = (-1/2, 1/2) # (min, max) tuple
            self.z_range_w = (-1/2, 1/2) # (min, max) tuple

            # initializers
            if self.coords_initializer == 'uniform':
                if self.coords_calib_size is None:
                    self.coords_calib_size = 0
                target_num_samples_h = self.z_dim_h - self.coords_calib_size
                target_num_samples_w = self.z_dim_w - self.coords_calib_size
                phi_h = np.random.uniform(low=self.z_range_h[0], high=self.z_range_h[1], size=[target_num_samples_h])
                phi_w = np.random.uniform(low=self.z_range_w[0], high=self.z_range_w[1], size=[target_num_samples_w])
                # add calib region
                phi_h = np.concatenate([phi_h, np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32)/self.h], axis=0)
                phi_w = np.concatenate([phi_w, np.arange(-self.coords_calib_size/2, self.coords_calib_size/2, dtype=np.int32)/self.w], axis=0)
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)

            self.phi_h = self.add_weight(name='phi_h',
                                        shape=[self.z_dim_h],
                                        dtype=tf.float32,
                                        initializer=var_init_h,
                                        trainable=self.optimize_sampling,
                                        constraint=ModuloConstraint(self.z_range_h[1]*2))
            self.phi_w = self.add_weight(name='phi_w',
                                        shape=[self.z_dim_w],
                                        dtype=tf.float32,
                                        initializer=var_init_w,
                                        trainable=self.optimize_sampling,
                                        constraint=ModuloConstraint(self.z_range_w[1]*2))


        # 2d case (optimization of horizontal k-space line locations)
        elif self.enc_arch =='matrix1d':
            self.z_dim = int(img_shape[-2]*self.undersamp_ratio)
            self.z_range_h = (-img_shape[-2]/2, img_shape[-2]/2) # (min, max) tuple
            self.z_range_w = (-img_shape[-1]/2, img_shape[-1]/2) # (min, max) tuple

            # initializers
            if self.coords_initializer == 'uniform':
                var_init_h = tf.random_uniform_initializer(
                    minval=self.z_range_h[0],
                    maxval=self.z_range_h[1]
                )
                var_init_w = tf.constant_initializer(
                    value=np.arange(*self.z_range_w)
                )
            elif self.coords_initializer == 'normal':
                # with truncated normal values more than
                # 2stddevs are discarded and resampled
                var_init_h = tf.keras.initializers.TruncatedNormal(
                    mean=0.0, stddev=self.z_range_h[1]/2
                )
                var_init_w = tf.constant_initializer(
                    value=np.arange(*self.z_range_w)
                )
            elif self.coords_initializer == 'mask':
                sampling_mask = np.load(self.fixed_mask_path)
                if sampling_mask.shape != (self.h, self.w):
                    raise ValueError('Mask shape and input shape are inconsistent.')
                phi_h_indices, phi_w_indices = np.nonzero(sampling_mask)
                phi_h = phi_h_indices - sampling_mask.shape[0]//2
                phi_w = phi_w_indices - sampling_mask.shape[1]//2
                var_init_h = tf.constant_initializer(value=phi_h)
                var_init_w = tf.constant_initializer(value=phi_w)

            self.phi_h = self.add_weight(name='phi_h',
                                        shape=[self.z_dim],
                                        dtype=tf.float32,
                                        initializer=var_init_h,
                                        trainable=self.optimize_sampling,
                                        constraint=ModuloConstraint(self.h))
            self.phi_w = self.add_weight(name='phi_w',
                                        shape=[self.w],
                                        dtype=tf.float32,
                                        initializer=var_init_w,
                                        trainable=False,
                                        constraint=ModuloConstraint(self.w))

        # fft case
        elif self.enc_arch == 'fft':
            self.accelereations = [1./undersamp_ratio for undersamp_ratio in self.undersamp_ratio]
            self.z_range_h = (-img_shape[-2]/2, img_shape[-2]/2) # (min, max) tuple
            self.z_range_w = (-img_shape[-1]/2, img_shape[-1]/2) # (min, max) tuple
            if self.fft_mask_type == 'fixed_mask':
                sampling_mask = np.load(self.fixed_mask_path)
                self.fft_mask_func = lambda *args, **kwargs: tf.convert_to_tensor(sampling_mask, dtype=tf.float32)[tf.newaxis]
            else:
                self.fft_mask_func = subsample.create_mask_for_mask_type(
                    self.fft_mask_type,
                    center_fractions=self.fft_mask_center_fractions,
                    calib_sizes=self.fft_mask_calib_sizes,
                    accelerations=self.accelereations,
                )


        # Decoding
        if self.dec_arch == 'unrolled':
            if self.unrolled_share_proxresblocks:
                prox_res_block = ProxResBlock(self.unrolled_num_resblocks,
                                              self.unrolled_num_filters,
                                              self.unrolled_kernel_size,
                                              self.unrolled_normalization_layer,
                                              self.unrolled_act_fun)
                self.prox_res_blocks = [prox_res_block for _ in range(self.unrolled_steps)]
            else:
                self.prox_res_blocks = [ProxResBlock(self.unrolled_num_resblocks,
                                                     self.unrolled_num_filters,
                                                     self.unrolled_kernel_size,
                                                     self.unrolled_normalization_layer,
                                                     self.unrolled_act_fun) for _ in range(self.unrolled_steps)]
            self.prox_step_sizes = [self.add_weight(name=f'prox_step_{step}',
                                                    shape=[],
                                                    initializer=tf.keras.initializers.Constant(-2.0),
                                                    dtype=tf.float32,
                                                    trainable=True) for step in range(self.unrolled_steps)]
        else:
            raise ValueError(f'Decoding architecture {self.dec_arch} not supported.')


    @tf.function
    def call(self, inputs, training=None):
        "Create model's logic; create output from the inputs. Assumes the inputs are NCHW"
        
        image, sensemap = inputs

        # encode input image
        temp_ops = self.encode(image, sensemap, training)
        mu_kspace = temp_ops.pop('mu_kspace')
        # sample latent variable
        kspace = self.sample_complex(mu_kspace, self.noise_std)
        # decode by reconstruction network
        reconstruction = self.recon_fn(kspace, sensemap, training, **temp_ops)
        return reconstruction


    def sample_complex(self, mean, std):
        """
        Sample from a Gaussian distibution with a given complex-valued mean and diagonal covariance.

        Inputs:
            mean: Complex valued mean tensor

        Returns:
            sample : Complex valued tensor of sampled from the specified Gaussian distribution.
        """

        mean_ch = mrutils.complex_to_channels(mean, axis=1)
        noise = tf.random.normal(tf.shape(mean_ch), 0, 1, dtype=tf.float32)
        sample_ch = tf.add(mean_ch, tf.multiply(std, noise))
        sample = mrutils.channels_to_complex(sample_ch, axis=1)
        return sample


    def encode_tfnufft(self, image, sensemap, training):
        """
        Performs NUFFT based encoding using tfNUFFT.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase (not used)

        Returns:
            A Python dict consisting of
            mu_kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z].
                       This corresponds to the mean tensor of the Gaussian distribution.
        """

        mu_kspace = self.tfnufft_forward(image, sensemap)
        return {'mu_kspace': mu_kspace}


    def encode_tfnufft_from_kspace(self, raw_kspace, sensemap, training):
        """
        Performs NUFFT based encoding using tfNUFFT starting from raw kspace data.
        Essentially performs interpolation in kspace.

        Inputs:
            raw_kspace: Complex valued fully sampled kspace tensor of shape [N, C, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W] (not used)
            training: Boolean value indicating training phase (not used)

        Returns:
            A Python dict consisting of
            mu_kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z].
                       This corresponds to the mean tensor of the Gaussian distribution.
        """

        coil_images = mrutils.ifft2c(raw_kspace)
        coords_h, coords_w = self.get_coordinates()
        coords = tf.stack([coords_h, coords_w], axis=-1)
        mu_kspace = nufft(coil_images, coords, self.tfnufft_oversamp, self.tfnufft_width)
        return {'mu_kspace': mu_kspace}


    def tfnufft_forward(self, image, sensemap):
        """
        Performs forward sense mapping using tfNUFFT.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]

        Returns:
            kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z]
        """

        image = tf.expand_dims(image, axis=-3)
        image = sensemap * image
        coords_h, coords_w = self.get_coordinates()
        coords = tf.stack([coords_h, coords_w], axis=-1)
        mu_kspace = nufft(image, coords, self.tfnufft_oversamp, self.tfnufft_width)
        return mu_kspace


    def tfnufft_transpose(self, kspace, sensemap, normalize=False):
        """
        Performs adjoint sense mapping using adjoint tfNUFFT.

        Inputs:
            kspace: Complex valued kspace tensor of shape [N, C, Z]
            sensemap: Sensitivity maps complex valued tensor of shape [N, C, H, W]
            normalize: Boolean value indicating whether normalization should be applied at the output

        Returns:
            image: Complex valued image tensor resulting from sense recon of shape [N, H, W]
        """

        # nufft adjoint
        coords_h, coords_w = self.get_coordinates()
        coords = tf.stack([coords_h, coords_w], axis=-1)
        oshape = [self.h, self.w]
        image = nufft_adjoint(kspace, coords, oshape, self.tfnufft_oversamp, self.tfnufft_width)
        # sense operator
        image = tf.math.conj(sensemap) * image
        image = tf.reduce_sum(image, axis=-3)
        if normalize:
            scale = tf.reduce_max(tf.abs(image), axis=[-2, -1], keepdims=True)
            scale = tf.cast(scale, tf.complex64)
            image = tf.math.divide_no_nan(image, scale)
        return image


    ## jmodl type of optimization functions
    def encode_matrix_jmodl_2d(self, image, sensemap, training):
        """
        Performs jmodl encoding using jmodl forward functions. Calculates the jmodl nudft 
        matrices and returns them in the return dict for future use.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase (not used)

        Returns:
            A Python dict consisting of
            mu_kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z].
                       This corresponds to the mean tensor of the Gaussian distribution.
            A_h: jmodl forward mapping matrix for the h dimension of shape [Z_H, H]
            A_w: jmodl forward mapping matrix for the w dimension of shape [Z_W, W]
        """

        h_range = tf.cast(tf.range(self.h), dtype=tf.complex64)
        w_range = tf.cast(tf.range(self.w), dtype=tf.complex64)

        jTwoPi = tf.constant(1j*2*math.pi, dtype=tf.complex64)
        scale_h = tf.constant(1./math.sqrt(self.h), dtype=tf.complex64)
        scale_w = tf.constant(1./math.sqrt(self.w), dtype=tf.complex64)

        phi_h = tf.cast(self.phi_h[:,tf.newaxis], tf.complex64)
        phi_w = tf.cast(self.phi_w[:,tf.newaxis], tf.complex64)

        A_h = tf.exp(-jTwoPi * phi_h * (h_range - self.h/2)) * scale_h  # shape [self.z_dim_h, h]
        A_w = tf.exp(-jTwoPi * phi_w * (w_range - self.w/2)) * scale_w  # shape [self.z_dim_w, w]

        mu_kspace = self.matrix_jmodl_2d_forward(image, sensemap, A_h, A_w)
        return {'mu_kspace': mu_kspace, 'A_h': A_h, 'A_w': A_w}


    def encode_matrix_jmodl_2d_from_kspace(self, raw_kspace, sensemap, training):
        """
        Performs NUFFT based encoding using tfNUFFT starting from raw kspace data.
        Essentially performs interpolation in kspace.
        Calculates the jmodl nudft matrices and returns them in the return dict for future use.

        Inputs:
            raw_kspace: Complex valued fully sampled kspace tensor of shape [N, C, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W] (not used)
            training: Boolean value indicating training phase (not used)

        Returns:
            A Python dict consisting of
            mu_kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z].
                       This corresponds to the mean tensor of the Gaussian distribution.
            A_h: jmodl forward mapping matrix for the h dimension of shape [Z_H, H]
            A_w: jmodl forward mapping matrix for the w dimension of shape [Z_W, W]
        """

        h_range = tf.cast(tf.range(self.h), dtype=tf.complex64)
        w_range = tf.cast(tf.range(self.w), dtype=tf.complex64)

        jTwoPi = tf.constant(1j*2*math.pi, dtype=tf.complex64)
        scale_h = tf.constant(1./math.sqrt(self.h), dtype=tf.complex64)
        scale_w = tf.constant(1./math.sqrt(self.w), dtype=tf.complex64)

        phi_h = tf.cast(self.phi_h[:,tf.newaxis], tf.complex64)
        phi_w = tf.cast(self.phi_w[:,tf.newaxis], tf.complex64)

        A_h = tf.exp(-jTwoPi * phi_h * (h_range - self.h/2)) * scale_h  # shape [self.z_dim_h, h]
        A_w = tf.exp(-jTwoPi * phi_w * (w_range - self.w/2)) * scale_w  # shape [self.z_dim_w, w]

        coil_images = mrutils.ifft2c(raw_kspace)
        mu_kspace = tf.matmul(A_h @ coil_images, A_w, transpose_b=True)
        return {'mu_kspace': mu_kspace, 'A_h': A_h, 'A_w': A_w}


    def matrix_jmodl_2d_forward(self, image, sensemap, A_h, A_w):
        """
        Performs forward sense mapping using jmodl type forward operators.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            A_h: jmodl forward mapping matrix for the h dimension of shape [Z_H, H]
            A_w: jmodl forward mapping matrix for the w dimension of shape [Z_W, W]

        Returns:
            kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z_H, Z_W]
        """

        image = tf.expand_dims(image, axis=-3)
        image = sensemap * image
        kspace = tf.matmul(A_h @ image, A_w, transpose_b=True)
        return kspace


    def matrix_jmodl_2d_transpose(self, kspace, sensemap, A_h, A_w):
        """
        Performs adjoint sense mapping using jmodl type adjoint operators.

        Inputs:
            kspace: Complex valued kspace tensor of shape [N, C, Z_H, Z_W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            A_h: jmodl forward mapping matrix for the h dimension of shape [Z_H, H]
            A_h: jmodl forward mapping matrix for the w dimension of shape [Z_W, W]

        Returns:
            image: Complex valued image tensor of shape [N, H, W]
        """

        image = tf.matmul(tf.transpose(A_h, conjugate=True) @ kspace, tf.transpose(A_w, conjugate=True), transpose_b=True)
        # sense operator
        image = tf.math.conj(sensemap) * image
        image = tf.reduce_sum(image, axis=-3)
        return image


    def encode_fft(self, image, sensemap, training):
        """
        Performs FFT based encoding.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase

        Returns:
            A Python dict consisting of
            mu_kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z].
                       This corresponds to the mean tensor of the Gaussian distribution.
            sampling_mask: Binary sampling mask tensor of shape [H, W] or [N, H, W]
        """
        
        mask_seed = self.fft_mask_train_seed if training else self.fft_mask_val_seed
        sampling_mask = self.fft_mask_func(shape=tf.shape(image), seed=mask_seed)
        self.current_mask = sampling_mask # TODO(calkan): find a better way to pass the mask to self.get_coordinates
        mu_kspace = self.fft_forward(image, sensemap, sampling_mask)
        return {'mu_kspace': mu_kspace, 'sampling_mask': sampling_mask}


    def encode_matrix_nudft(self, image, sensemap, training):
        """
        Performs matrix based NUDFT encoding (NUDFT Type I).

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase (not used)

        Returns:
            A Python dict consisting of 
            mu_kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z].
                       This corresponds to the mean tensor of the Gaussian distribution.
            nudft_mtrx: NUDFT Type I matrix that maps uniform image domain samples to
                        non-uniform k-space domain samples. Has shape [Z, H, W].
            
        """

        # calculate nufft matrix 
        # (non-uniform frequencies are in [-1/2,1/2] range, uniform image domain samples are in [0,H] or [0,W] range)
        nudft_mtrx_h = tf.tensordot(self.phi_h/self.h, tf.range(self.h, dtype=tf.float32), axes=0) # [Z,H]
        nudft_mtrx_w = tf.tensordot(self.phi_w/self.w, tf.range(self.w, dtype=tf.float32), axes=0) # [Z,W]
        nudft_mtrx = tf.expand_dims(nudft_mtrx_h, axis=-1) + tf.expand_dims(nudft_mtrx_w, axis=-2)
        nudft_mtrx = tf.exp(tf.complex(0., -2*math.pi*nudft_mtrx)) # [Z,H,W]

        # compute NUDFT
        mu_kspace = self.matrix_nudft_forward(image, sensemap, nudft_mtrx)
        return {'mu_kspace': mu_kspace, 'nudft_mtrx': nudft_mtrx}
    

    def encode_matrix_nudft1d(self, image, sensemap, training):
        """
        Performs matrix based 1d-NUDFT encoding (NUDFT Type I).
        This function first applies standard FFT along the frequency encoding (width) dimension.
        Then applies NUDFT encoding along non-uniformly sampled phase encoding (height) dimension.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase (not used)

        Returns:
            A Python dict consisting of 
            mu_kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z].
                       This corresponds to the mean tensor of the Gaussian distribution.
            nudft_mtrx: NUDFT Type I matrix that maps uniform image domain samples to
                        non-uniform k-space domain samples. Has shape [Z, H, W].
            
        """

        # calculate 1d nufft matrix 
        # (non-uniform frequencies are in [-1/2,1/2] range, uniform image domain samples are in [0,H] or [0,W] range)
        nudft_mtrx = tf.tensordot(self.phi_h/self.h, tf.range(self.h, dtype=tf.float32), axes=0) # [Z,H]
        nudft_mtrx = tf.exp(tf.complex(0., -2*math.pi*nudft_mtrx)) # [Z,H]

        # compute NUDFT
        mu_kspace = self.matrix_nudft1d_forward(image, sensemap, nudft_mtrx)
        return {'mu_kspace': mu_kspace, 'nudft_mtrx': nudft_mtrx}        


    def fft_forward(self, image, sensemap, sampling_mask):
        """
        Performs forward sense mapping using FFT.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            sampling_mask: Binary sampling mask tensor of shape broadcastable to [N, H, W]
                where one of H or W can be 1 for 2D sampling masks


        Returns:
            kspace: Complex valued tensor of kspace values for each coil of shape [N, C, H, W]
        """

        image = tf.expand_dims(image, axis=-3)
        image = sensemap * image
        kspace = mrutils.fft2c(image)
        # apply sampling mask
        kspace = tf.expand_dims(tf.cast(sampling_mask, tf.complex64), axis=-3) * kspace
        return kspace
    

    def fft_transpose(self, kspace, sensemap, **kwargs):
        """
        Performs adjoint sense mapping using IFFT.

        Inputs:
            kspace: Complex valued kspace tensor of shape [N, C, H, W]
            sensemap: Sensitivity maps complex valued tensor of shape [N, C, H, W]

        Returns:
            image: Complex valued image tensor resulting from sense recon of shape [N, H, W]
        """

        image = mrutils.ifft2c(kspace)
        image = tf.math.conj(sensemap) * image
        image = tf.reduce_sum(image, axis=-3)
        return image


    def matrix_nudft_forward(self, image, sensemap, nudft_mtrx):
        """
        Performs forward sense mapping using matrix based NUDFT (Type I).

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            nudft_mtrx: NUDFT matrix that maps uniform image domain samples to
                        non-uniform k-space domain samples. Has shape [Z, H, W].

        Returns:
            kspace: Complex valued tensor of kspace values for each coil of shape [N, C, Z]
        """

        image = tf.expand_dims(image, axis=-3)
        image = sensemap * image
        # fftshift before transform
        image = tf.signal.fftshift(image, axes=(2, 3))
        kspace = tf.tensordot(image, nudft_mtrx, axes=[[-2, -1], [-2, -1]])
        return kspace


    def matrix_nudft_transpose(self, kspace, sensemap, nudft_mtrx, normalize=True):
        """
        Performs adjoint sense mapping using matrix based NUDFT (Type II).

        Inputs:
            kspace: Complex valued kspace tensor of shape [N, C, Z]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            nudft_mtrx: NUDFT matrix that maps uniform image domain samples to
                        non-uniform k-space domain samples. Has shape [Z, H, W].
            normalize: Boolean value indicating whether normalization should be applied at the output

        Returns:
            image: Complex valued image tensor resulting from sense recon of shape [N, H, W]
        """

        image = tf.tensordot(kspace, tf.math.conj(nudft_mtrx), axes=[[-1], [0]]) # [N,C,H,W]
        # reverse fftshift
        image = tf.signal.ifftshift(image, axes=(2,3))
        image = tf.math.conj(sensemap) * image
        image = tf.reduce_sum(image, axis=-3)
        if normalize:
            scale = tf.reduce_max(tf.abs(image), axis=[-2, -1], keepdims=True)
            scale = tf.cast(scale, tf.complex64)
            image = image/scale
        return image
    

    def matrix_nudft1d_forward(self, image, sensemap, nudft_mtrx):
        """
        Performs forward sense mapping using matrix based 1d-NUDFT (NUDFT Type I).
        This function first applies standard FFT along the frequency encoding (width) dimension.
        Then applies NUDFT encoding along non-uniformly sampled phase encoding (height) dimension.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            nudft_mtrx: 1d NUDFT matrix that maps uniform image domain samples along H dimension to
                        non-uniform k-space domain samples. Has shape [Z, H].

        Returns:
            kspace: Complex valued tensor of kspace values for each coil of shape [N, C, W, Z]
        """

        # perform 1d fft along freq encoding first
        image = tf.expand_dims(image, axis=-3)
        image = sensemap * image # [N, C, H, W]
        kspace = mrutils.fft1c(image) # [N, C, H, W]
        # then apply matrix nudft 1d in phase encoding direction
        # fftshift before transform
        kspace = tf.signal.fftshift(kspace, axes=[2])
        kspace = tf.tensordot(kspace, nudft_mtrx, axes=[[-2], [-1]]) # [N, C, W, Z]
        return kspace


    def matrix_nudft1d_transpose(self, kspace, sensemap, nudft_mtrx, normalize=True):
        """
        Performs adjoint sense mapping using matrix based 1d-NUDFT (NUDFT Type II).
        This function first applies adjoint NUDFT along non-uniformly sampled phase encoding (height) dimension.
        Then applies standard iFFT along the frequency encoding (width) dimension.

        Inputs:
            kspace: Complex valued kspace tensor of shape [N, C, W, Z]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            nudft_mtrx: 1d NUDFT matrix that maps uniform image domain samples along H dimension to
                        non-uniform k-space domain samples. Has shape [Z, H].
            normalize: Boolean value indicating whether normalization should be applied at the output

        Returns:
            kspace: Complex valued image tensor resulting from sense recon of shape [N, H, W]
        """

        image = tf.tensordot(kspace, tf.math.conj(nudft_mtrx), axes=[[-1], [0]]) # [N,C,W, H]
        image = tf.transpose(image, perm=[0, 1, 3, 2]) # [N,C,H,W]
        # reverse fftshift along H
        image = tf.signal.ifftshift(image, axes=[2])
        image = mrutils.ifft1c(image) # [N,C,H,W]
        image = tf.math.conj(sensemap) * image
        image = tf.reduce_sum(image, axis=-3)
        if normalize:
            scale = tf.reduce_max(tf.abs(image), axis=[-2, -1], keepdims=True)
            scale = tf.cast(scale, tf.complex64)
            image = tf.math.divide_no_nan(image, scale)
        return image


    def unrolled_prox(self, kspace, sensemap, training, **kwargs):
        """
        Performs unrolled proximal gradient descent reconstruction

        Inputs:
            kspace: Complex valued kspace tensor of shape [N, C, Z]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase
            nudft_mtrx: (optional) NUDFT matrix that maps uniform image domain samples to
                        non-uniform k-space domain samples. Has shape [Z, H, W].
                        Used only if enc_arch is 'matrix'.

        Returns:
            image: Complex valued image tensor resulting from unrolled recon of shape [N, H, W]
        """

        # compute zero-filled recon
        zf_image = self.adjoint(kspace, sensemap, **kwargs)
        # initialize optimization variable with zero-filled recon
        image = zf_image
        for prox_res_block, prox_step_size in zip(self.prox_res_blocks, self.prox_step_sizes):
            grad_update = self.adjoint(self.forward(image, sensemap, **kwargs), sensemap, **kwargs) - zf_image
            image = image + tf.cast(prox_step_size, tf.complex64)*grad_update
            image = mrutils.complex_to_channels(image, axis=1) # channels first
            image = prox_res_block(image, training)
            image = mrutils.channels_to_complex(image, axis=1) # channels first
        return image


    def unet_recon(self, kspace, sensemap, training, **kwargs):
        """
        Performs unet reconstruction

        Inputs:
            kspace: Complex valued kspace tensor of shape [N, C, Z]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase
            nudft_mtrx: (optional) NUDFT matrix that maps uniform image domain samples to
                        non-uniform k-space domain samples. Has shape [Z, H, W].
                        Used only if enc_arch is 'matrix'.
                        
        Returns:
            image: Complex valued image tensor resulting from unrolled recon of shape [N, H, W]
        """

        # compute zero-filled recon
        zf_image = self.adjoint(kspace, sensemap, **kwargs)
        zf_image = tf.math.divide_no_nan(zf_image, 
                                         tf.cast(tf.reduce_max(tf.abs(zf_image), axis=[1,2], keepdims=True), zf_image.dtype))
        # split into real and imaginary
        zf_image = mrutils.complex_to_channels(zf_image, axis=1) # channels first
        image = self.unet(zf_image, training=training)
        # combine real and imaginary
        image = mrutils.channels_to_complex(image, axis=1) # chanels first
        return image


    def zerofill_recon(self, image, sensemap, training=None):
        """
        Performs zero-filled reconstruction on given input image.
        It performs forward_sense encoding followed by adjoint_sense decoding.

        Inputs:
            image: Complex valued image tensor of shape [N, H, W]
            sensemap: Complex valued sensitivity maps tensor of shape [N, C, H, W]
            training: Boolean value indicating training phase

        Returns:
            image: Complex valued image tensor resulting from nufft recon of shape [N, H, W]
        """

        # encode input image with nufft
        temp_ops = self.encode(image, sensemap, training)
        kspace = temp_ops.pop('mu_kspace')
        # decode by adjoint_nufft
        reconstruction = self.adjoint(kspace, sensemap, **temp_ops)
        return reconstruction
    

    def get_trainable_variables(self, layer_type):
        """
        Return the trainable variables for encoder or decoder.

        Inputs:
            layer_type: one of 'encoder' or 'decoder' specifying from which layer to return the variables.

        Returns:
            variable_list: List of trainable variables from the layer specified by `layer_type`.
        """

        if layer_type == 'encoder':
            variable_list = [var for var in self.trainable_variables if 'phi' in var.name]
        elif layer_type == 'decoder':
            variable_list = [var for var in self.trainable_variables if 'phi' not in var.name]
        else:
            raise ValueError('`layer_type` can only be ''encoder'' or ''decoder''')
        
        return variable_list
        

    def point_spread_function(self, normalize=True, shift=[0,0]):
        """
        Calculates the point spread function of the trajectory defined by the model.
        It performs forward nuFFT followed by adjoint nuFFT on an impulse object centered
        at the coordinates specified by the shift parameter.

        Inputs:
            normalize: Boolean value indicating whether normalization should be applied at the output 
                       of adjoint nuFFT.
            shift: List or tuple of two entries specifying the coordinates of the input impulse object.

        Returns:
            psf: Complex valued image tensor of shape [H,W] corresponding to the point spread function
                 of the trajectory.
        """
        
        psf_input_image = np.zeros([1, self.h, self.w], dtype=np.complex64)
        psf_input_image[:,0,0] = 1
        psf_input_image = tf.signal.fftshift(psf_input_image)
        psf_input_image = tf.roll(psf_input_image, shift=shift, axis=[1,2])

        coords_h, coords_w = self.get_coordinates()
        coords = tf.stack([coords_h, coords_w], axis=-1)
        kspace = nufft(psf_input_image, coords, self.tfnufft_oversamp, self.tfnufft_width)
        oshape = [self.h, self.w]
        psf = nufft_adjoint(kspace, coords, oshape, self.tfnufft_oversamp, self.tfnufft_width)
        if normalize:
            scale = tf.reduce_max(tf.abs(psf), axis=[-2, -1], keepdims=True)
            scale = tf.cast(scale, tf.complex64)
            psf = psf/scale
        return psf[0]
    

    def point_spread_function_v2(self, normalize=True):
        """
        Calculates the point spread function of the trajectory defined by the model.
        It performs adjoint nuFFT on a set of impulses at the coordinates specified by
        self.get_coordinates().

        Inputs:
            normalize: Boolean value indicating whether normalization should be applied at the output 
                       of adjoint nuFFT.

        Returns:
            psf: Complex valued image tensor of shape [H,W] corresponding to the point spread function
                 of the trajectory.
        """
        
        coords_h, coords_w = self.get_coordinates()
        coords = tf.stack([coords_h, coords_w], axis=-1)
        oshape = [self.h, self.w]
        kspace = tf.cast(tf.ones([1, self.z_dim]), dtype=tf.complex64)
        psf = nufft_adjoint(kspace, coords, oshape, self.tfnufft_oversamp, self.tfnufft_width)
        if normalize:
            scale = tf.reduce_max(tf.abs(psf), axis=[-2, -1], keepdims=True)
            scale = tf.cast(scale, tf.complex64)
            psf = psf/scale
        return psf[0]


    def plot_coords(self):
        """
        Returns a matplotlib figure containing encoded coordinates.

        Returns:
            figure: matplotlib.pyplot figure containing encoded coordinates
        """
        
        # no need to use GPU memory for this
        with tf.device('/CPU:0'):
            coords_h, coords_w = self.get_coordinates()
        figure, axes = plt.subplots(1, 1, figsize=(10,10))
        axes.plot(coords_w.numpy(), coords_h.numpy(), '.')
        # TODO(calkan): find a solution that removes the if condition here
        if self.enc_arch == 'matrix_jmodl_2d':
            axes.set_xlim(np.array(self.z_range_w) * self.w)
            axes.set_ylim(np.array(self.z_range_h) * self.h)
        else:
            axes.set_xlim(self.z_range_w)
            axes.set_ylim(self.z_range_h)
        # axes.axis('equal')
        return figure


    def plot_psfs(self):
        """
        Returns a matplotlib figure containing point spread functions.

        Returns:
            figure: matplotlib.pyplot figure containing point spread functions
        """
        
        psf = self.point_spread_function(normalize=False)
        imv = np.abs(psf)
        imv = np.log10(imv, out=np.ones_like(imv) * -31, where=imv != 0)
        figure, axes = plt.subplots(1, 1, figsize=(6*self.w/self.h,6))
        pcm = axes.imshow(imv, vmin=np.floor(imv.min()), vmax=imv.max(), cmap='gray')
        axes.set_xticks([])
        axes.set_yticks([])
        plt.colorbar(pcm, ax=axes)
        axes.axis('off')
        return figure


    def plot_coords_and_psfs(self):
        """
        Returns a matplotlib figure containing encoded coordinates and point spread functions.

        Returns:
            figure: matplotlib.pyplot figure containing encoded coordinates and point spread functions
        """
        
        # no need to use GPU memory for this
        with tf.device('/CPU:0'):
            coords_h, coords_w = self.get_coordinates()
        psf = self.point_spread_function(normalize=False)
        imv = np.abs(psf)
        imv = np.log10(imv, out=np.ones_like(imv) * -31, where=imv != 0)
        # figure, axes = plt.subplots(2, 1, figsize=(6*self.w/self.h,6*2))
        figure, axes = plt.subplots(2, 1, figsize=(10,20))
        axes[0].plot(coords_w.numpy(), coords_h.numpy(), '.')
        # TODO(calkan): find a solution that removes the if condition here
        if self.enc_arch == 'matrix_jmodl_2d':
            axes[0].set_xlim(np.array(self.z_range_w) * self.w)
            axes[0].set_ylim(np.array(self.z_range_h) * self.h)
        else:
            axes[0].set_xlim(self.z_range_w)
            axes[0].set_ylim(self.z_range_h)
        # axes[0].axis('equal')

        # pcm = axes[1].imshow(imv, vmin=np.floor(imv.min()), vmax=imv.max(), cmap='gray')
        pcm = axes[1].imshow(imv, vmin=np.floor(np.percentile(imv,1)), vmax=imv.max(), cmap='gray')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(pcm, ax=axes[1])
        axes[1].axis('off')

        plt.tight_layout()
        return figure

    
    def get_coordinates(self):
        """
        Returns the k-space coordinates defined by phi_h and phi_w.

        Returns:
            coords_h: k_y coordinates
            coords_w: k_x coordinates
        """

        if self.enc_arch == 'matrix' or self.enc_arch == 'tfnufft':
            coords_h = self.phi_h
            coords_w = self.phi_w
        elif self.enc_arch == 'matrix_jmodl_2d':
            coords_h = tf.repeat(self.phi_h, repeats=[self.z_dim_w]*self.z_dim_h) * self.h
            coords_w = tf.tile(self.phi_w, multiples=[self.z_dim_h]) * self.w
        elif self.enc_arch == 'nufft1d' or self.enc_arch == 'matrix1d':
            coords_h = tf.repeat(self.phi_h, repeats=[self.w]*self.z_dim)
            coords_w = tf.tile(self.phi_w, multiples=[self.z_dim])
        elif self.enc_arch == 'fft': # shows only the first mask in the batch if batch_size>1
            mask = self.current_mask[0]
            nonzero_indices = tf.where(mask)
            phi_h_indices = nonzero_indices[:,0] 
            phi_w_indices = nonzero_indices[:,1]
            coords_h = phi_h_indices - mask.shape[0]//2
            coords_w = phi_w_indices - mask.shape[1]//2
        else:
            coords_h = self.phi_h
            coords_w = self.phi_w

        return coords_h, coords_w
    

    def get_num_samples(self):
        """
        Returns the number of samples in k-space.

        Returns:
            num_samples: number of samples in k-space
        """

        if self.enc_arch == 'matrix' or self.enc_arch == 'tfnufft':
            num_samples = self.z_dim
        elif self.enc_arch == 'matrix_jmodl_2d':
            num_samples = self.z_dim
        elif self.enc_arch == 'matrix1d':
            num_samples = self.z_dim * self.w
        else:
            num_samples = self.z_dim

        return num_samples


    def get_config(self):
        config = super(MR_UAE, self).get_config()
        config.update({
            'undersamp_ratio': self.undersamp_ratio,
            'noise_std': self.noise_std,
            'dec_arch': self.dec_arch,
            'enc_arch': self.enc_arch,
        })
        if self.dec_arch == 'unrolled':
            config.update({
                'unrolled_steps': self.unrolled_steps,
                'unrolled_num_resblocks': self.unrolled_num_resblocks,
                'unrolled_num_filters': self.unrolled_num_filters,
                'unrolled_kernel_size': self.unrolled_kernel_size,
                'unrolled_normalization_layer': self.unrolled_normalization_layer,
                'unrolled_act_fun': self.unrolled_act_fun
            })
        elif self.dec_arch == 'unet':
            config.update({
                'unet_num_features': self.unet_num_features,
                'unet_num_levels': self.unet_num_levels,
                'unet_normalization_layer': self.unet_normalization_layer,
                'unet_act_fun': self.unet_act_fun
            })
        
        return config


class Conv2DBlock(tf.keras.layers.Layer):
    """
    2D convolutional block that has customizable normalizaiton and activation layers.
    Assumes channels_first ordering.
    """

    def __init__(self, num_filters, kernel_size, normalization, activation):
        super(Conv2DBlock, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation

        # conv
        self.conv_layer = tf.keras.layers.Conv2D(num_filters, 
                                                kernel_size=kernel_size,
                                                padding='same',
                                                data_format='channels_first')

        # normalization
        if self.normalization == 'instance':
            self.norm_layer = tfa.layers.InstanceNormalization(axis=1) # channels first
        elif self.normalization == 'layer':
            self.norm_layer = tf.keras.layers.LayerNormalization(axis=[1,2,3]) # tfa vs tf convention is slightly different
        elif self.normalization == 'batch':
            self.norm_layer = tf.keras.layers.BatchNormalization(axis=1) # channels first
        else:
            self.norm_layer = tf.keras.layers.Lambda(lambda x: tf.identity(x)) # no normalization layer

        # act
        if self.activation == 'lrelu':
            self.act_layer = tf.keras.layers.LeakyReLU()
        elif self.activation == 'relu':
            self.act_layer = tf.keras.layers.ReLU()
        else:
            self.act_layer = tf.keras.layers.Lambda(lambda x: tf.identity(x)) # no activation layer
    
    def call(self, inputs, training=None):
        # conv
        result = self.conv_layer(inputs, training=training)
        # norm
        result = self.norm_layer(result, training=training)
        # act
        result = self.act_layer(result, training=training)
        return result

    def get_config(self):
        config = super(Conv2DBlock, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation
        })
        return config


class TransposeConv2DBlock(tf.keras.layers.Layer):
    """
    2D transpose convolutional block that has customizable normalization and activation layers.
    Assumes channels_first ordering.
    """

    def __init__(self, num_filters, kernel_size, strides, normalization, activation):
        super(TransposeConv2DBlock, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.normalization = normalization
        self.activation = activation
        
        # conv transpose
        self.t_conv_layer = tf.keras.layers.Conv2DTranspose(num_filters, 
                                                            strides=strides,
                                                            kernel_size=kernel_size,
                                                            padding='same',
                                                            data_format='channels_first')

        # normalization
        if self.normalization == 'instance':
            self.norm_layer = tfa.layers.InstanceNormalization(axis=1) # channels first
        elif self.normalization == 'layer':
            self.norm_layer = tf.keras.layers.LayerNormalization(axis=[1,2,3]) # tfa vs tf convention is slightly different
        elif self.normalization == 'batch':
            self.norm_layer = tf.keras.layers.BatchNormalization(axis=1) # channels first
        else:
            self.norm_layer = tf.keras.layers.Lambda(lambda x: tf.identity(x)) # no normalization layer

        # act
        if self.activation == 'lrelu':
            self.act_layer = tf.keras.layers.LeakyReLU()
        elif self.activation == 'relu':
            self.act_layer = tf.keras.layers.ReLU()
        else:
            self.act_layer = tf.keras.layers.Lambda(lambda x: tf.identity(x)) # no activation layer
    
    def call(self, inputs, training=None):
        # conv
        result = self.t_conv_layer(inputs, training=training)
        # norm
        result = self.norm_layer(result, training=training)
        # act
        result = self.act_layer(result, training=training)
        return result

    def get_config(self):
        config = super(TransposeConv2DBlock, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'normalization': self.normalization,
            'activation': self.activation
        })
        return config


class ResBlock(tf.keras.layers.Layer):
    """
    Residual block consisting of two 2D convolutional layers each followed by
    normalization and activation layers. The block has a skip connection that
    ensures the input and output dimensions are consistent.
    Assumes channels_first ordering.
    """
    def __init__(self, num_filters, kernel_size, normalization, activation):
        super(ResBlock, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        
    def build(self, input_shape):
        input_shape_list = input_shape.as_list()
        self.conv1 = Conv2DBlock(self.num_filters,
                                self.kernel_size,
                                self.normalization, 
                                self.activation)
        self.conv2 = Conv2DBlock(self.num_filters,
                                self.kernel_size,
                                self.normalization,
                                self.activation)
        if input_shape_list[1] != self.num_filters:
            # 1x1 conv in skip connection for making the input and output dimensions the same
            self.shortcut = tf.keras.layers.Conv2D(self.num_filters, 
                                                    kernel_size=1,
                                                    padding='same',
                                                    data_format='channels_first')
        else:
            # direct skip connection
            self.shortcut = tf.keras.layers.Lambda(lambda x: tf.identity(x))

    def call(self, inputs, training=None):
        result = self.conv1(inputs, training=training)
        result = self.conv2(result, training=training)
        return result + self.shortcut(inputs, training=training)

    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation
            })
        return config


class ProxResBlock(tf.keras.layers.Layer):
    """
    Proximal residual block consisting of a given number of ResBlocks.
    Takes and returns real valued input and output, respectively.
    Assumes channels_first ordering.
    """
    def __init__(self, num_resblocks, num_filters, kernel_size, normalization, activation):
        super(ProxResBlock, self).__init__()
        self.num_resblocks = num_resblocks
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation

    def build(self, input_shape):
        input_shape_list = input_shape.as_list()
        dim_ch = input_shape_list[1]
        layers = [ResBlock(self.num_filters, self.kernel_size, self.normalization, self.activation)
                    for _ in range(self.num_resblocks)]
        # add final conv layer to downsample to input channel size for residual connection
        layers.append(tf.keras.layers.Conv2D(dim_ch,
                                        kernel_size=1,
                                        padding='same',
                                        data_format='channels_first'))
        self.layers = tf.keras.Sequential(layers)
    
    def call(self, inputs, training=None):
        result = self.layers(inputs, training=training) + inputs
        return result

    def get_config(self):
        config = super(ProxResBlock, self).get_config()
        config.update({
            'num_resblocks': self.num_resblocks,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation
        })
        return config


class Unet(tf.keras.Model):
    """
    Unet implementation.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015..

    Inputs:
        out_channels: Number of channels at the Unet output.
        num_pool_layers:  Number of down-sampling and up-sampling layers.
        channels: Number of output channels for the first convolution layer.
            At all subsequent depths, the number of output channels is doubled.
            For example, if `channels=32`, number of channels will be
            ``[32, 64, 128, ...]``.
            If `channels=None`, the channel size of the input will be used.
        kernel_size: Convolutional kernel width.
        normalization: Normalization type to use. Must be one of `instance`, `layer`,
            `batch` or None. None corresponds to no normalization.
        activation: Activation function to use. Must be one of `relu` or `lrelu`.
    """

    def __init__(self, out_channels, num_pool_layers, channels=None, kernel_size=3, normalization='instance', activation='lrelu', **kwargs):
        super(Unet, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.num_pool_layers = num_pool_layers
        self.channels = channels
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation

    def build(self, input_shape):
        input_shape_list = input_shape.as_list()
        dim_ch = input_shape_list[1]

        if self.channels is None:
            self.channels = dim_ch

        ch = self.channels
        self.down_sample_layers = [tf.keras.Sequential(
            [
                Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
                Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
            ]
        )]
        for _ in range(self.num_pool_layers-1):
            ch = ch * 2
            self.down_sample_layers.append(tf.keras.Sequential([
                tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid', data_format='channels_first'),
                Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
                Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
            ]))
        ch = ch * 2
        self.bottleneck_layer = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid', data_format='channels_first'),
            Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
        ])
        
        self.up_sample_layers = [tf.keras.Sequential(
            [TransposeConv2DBlock(ch // 2, self.kernel_size, 2, self.normalization, self.activation)]
        )]
        ch = ch // 2
        for _ in range(self.num_pool_layers-1):
            self.up_sample_layers.append(tf.keras.Sequential([
                Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
                TransposeConv2DBlock(ch // 2, self.kernel_size, 2, self.normalization, self.activation),
            ]))
            ch = ch // 2

        self.final_conv_layer = tf.keras.Sequential([
            Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
            Conv2DBlock(ch, self.kernel_size, self.normalization, self.activation),
            Conv2DBlock(self.out_channels, 1, self.normalization, self.activation),
        ])
    
    @tf.function
    def call(self, inputs, training=None):

        # apply down-sampling layers
        skip_connection_stack = []
        output = inputs
        for layer in self.down_sample_layers:
            output = layer(output, training=training)
            skip_connection_stack.append(output)
        
        # apply bottleneck layer
        output = self.bottleneck_layer(output, training=training)

        # apply up-sampling layers with skip connections
        for layer in self.up_sample_layers:
            skip_connection = skip_connection_stack.pop()
            upsample_output = layer(output, training=training)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            paddings = [[0,0],[0,0],[0,0],[0,0]]
            # paddings = tf.zeros([4,2], dtype=tf.int32)
            if tf.shape(skip_connection)[2] != tf.shape(upsample_output)[2]:
                # paddings = tf.tensor_scatter_nd_update(paddings, [[2,1]], [1])
                paddings[2][1] = 1 # padding bottom
            if tf.shape(skip_connection)[3] != tf.shape(upsample_output)[3]:
                # paddings = tf.tensor_scatter_nd_update(paddings, [[3,1]], [1])
                paddings[3][1] = 1 # padding right
            upsample_output = tf.pad(upsample_output, paddings, "REFLECT")

            # apply skip connection
            output = tf.concat([skip_connection, upsample_output], axis=1)

        # apply final conv layer
        output = self.final_conv_layer(output, training=training)
        return output

    def get_config(self):
        config = super(Unet, self).get_config()
        config.update({
            'out_channels': self.out_channels,
            'num_pool_layers': self.num_pool_layers,
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation
        })
        return config


class ModuloConstraint(tf.keras.constraints.Constraint):
    '''
    Constraint that enforces the weights to be in the interval (-modval/2.0, modval/2.0)
    '''
    def __init__(self, modval):
        super(ModuloConstraint, self).__init__()
        self.modval = modval

    def __call__(self, w):
        # return tf.math.floormod(w-self.modval/2.0, self.modval) - self.modval/2.0
        return self.floormod(w - self.modval/2.0) - self.modval/2.0

    def floormod(self, w):
        # tf.math.floormod fails in gpu graph mode for some reason. we are writing our own
        quotient = tf.math.floordiv(w, self.modval)
        remainder = w - quotient * self.modval
        return remainder

    def get_config(self):
        return {'modval': self.modval}

