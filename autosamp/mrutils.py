from absl import app
from absl import flags

import math
import numpy as np
import os
import glob
import tensorflow as tf

def channels_to_complex(image, axis, expand_dims=True):
    """
    Converts data from channels to complex.

    Inputs:
        image: Tensor consisting of real and imaginary channels.
        axis: Axis where the real and imaginary channels are stored.
        expand_dims: Indicates whether real/imaginary channels are stored
            in a new dimension or not.

    Returns:
        image_out: Complex valued output tensor.
    """

    if expand_dims:
        image_real, image_imag = tf.unstack(image, 2, axis=axis)
        image_out = tf.complex(image_real, image_imag)
    else:
        image_real, image_imag = tf.split(image, num_or_size_splits=2, axis=axis)
        image_out = tf.complex(image_real, image_imag)
    return image_out

def complex_to_channels(image, axis, expand_dims=True):
    """
    Converts data from complex to channels.

    Inputs:
        image: Complex valued tensor.
        axis: Axis where the real and imaginary channels will be stored.
        expand_dims: Indicates whether real/imaginary channels will be stored 
            in a new dimension or not.

    Returns:
        image_out: Real valued tensor with real/imaginary channels stored in 'axis'.
    """

    if expand_dims:
        image_out = tf.stack((tf.math.real(image), tf.math.imag(image)), axis=axis)
    else:
        image_out = tf.concat((tf.math.real(image), tf.math.imag(image)), axis=axis)
    return image_out

def kstoim_with_channels(kspace):
    "Converts k-space data to image data by direct IFFT on each channel."
    raise NotImplementedError

def fft2c(image):
    "Calculates centered orthonormal 2d FFT of innermost dimensions."
    num_elems = tf.shape(image)[-1] * tf.shape(image)[-2]
    scale = tf.sqrt(tf.cast(num_elems, tf.complex64))
    image_out = tf.signal.fftshift(image, axes=(-2, -1))
    image_out = tf.signal.fft2d(image_out) / scale
    image_out = tf.signal.ifftshift(image_out, axes=(-2, -1))
    return image_out

def ifft2c(image):
    "Calculates centered orthonormal 2d IFFT of innermost dimensions."
    num_elems = tf.shape(image)[-1] * tf.shape(image)[-2]
    scale = tf.sqrt(tf.cast(num_elems, tf.complex64))
    image_out = tf.signal.fftshift(image, axes=(-2, -1))
    image_out = tf.signal.ifft2d(image_out) * scale
    image_out = tf.signal.ifftshift(image_out, axes=(-2, -1))
    return image_out

def fft1c(image):
    "Calculates centered orthonormal 1d FFT of innermost dimmension."
    num_elems = tf.shape(image)[-1]
    scale = tf.sqrt(tf.cast(num_elems, tf.complex64))
    image_out = tf.signal.fftshift(image, axes=[-1])
    image_out = tf.signal.fft(image_out) / scale
    image_out = tf.signal.ifftshift(image_out, axes=[-1])
    return image_out

def ifft1c(image):
    "Calculates centered orthonormal 1d IFFT of innermost dimensions."
    num_elems = tf.shape(image)[-1]
    scale = tf.sqrt(tf.cast(num_elems, tf.complex64))
    image_out = tf.signal.fftshift(image, axes=[-1])
    image_out = tf.signal.ifft(image_out) * scale
    image_out = tf.signal.ifftshift(image_out, axes=[-1])
    return image_out

def rss(image, axis, keepdims=False):
    "Calculates root sum of squares reconstruction where coil dim is specified by 'axis'"
    image_out = tf.square(tf.abs(image))
    image_out = tf.reduce_sum(image_out, axis=axis, keepdims=keepdims)
    image_out = tf.sqrt(image_out)
    return image_out

def sense_forward(image, sensemap, channel_axis=-3):
    """
    Performs forward sense mapping using fft.

    Inputs:
        image: Image tensor without coil dimension, with shape [...,(T),H,W] 
        sensemap: Sensitivity maps tensor with coil dimension in 'channel_axis', has shape [...,C,(T),H,W]
        channel_axis: Coil sensitivity axis in 'sensemap'

    Returns:
        kspace: Tensor of kspace values for each coil with coil dimension in 'channel_axis'
            Has shape [...,C,(T),H,W]
    """

    image = tf.expand_dims(image, axis=channel_axis)
    image = sensemap * image
    kspace = fft2c(image)
    return kspace

def sense_transpose(kspace, sensemap, channel_axis=-3):
    """
    Performs transpose sense mapping using ifft.

    Inputs:
        kspace: kspace tensor with coil dimension in 'channel_axis', with shape [...,C,(T),H,W]
        sensemap: Sensitivity maps tensor with coil dimension in 'channel_axis', has shape [...,C,(T),H,W]
        channel_axis: Coil sensitivity axis in 'kspace' and 'sensemap'

    Returns:
        image: Image tensor resulting from sense recon of shape [...,(T),H,W]
    """

    image = ifft2c(kspace)
    image = tf.math.conj(sensemap) * image
    image = tf.reduce_sum(image, axis=channel_axis)
    return image

@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ASSERT_STATEMENTS)
def center_crop(image, crop_shape, data_format='channels_first'):
    """
    Apply center crop to a batch of images.

    Inputs:
        image: 3D or 4D Tensor to be center cropped.
        crop_shape: List of shape [h_out, w_out] specifying output shape.
        data_format: Data format of input and output data. Must be 'channels_first' or 'channels_last'.
    """

    if data_format == 'channels_first':
        offset_height = (tf.shape(image)[-2] - crop_shape[0]) // 2
        offset_width = (tf.shape(image)[-1] - crop_shape[1]) // 2
        assert (offset_height >= 0) & (offset_width >= 0)
        image = image[..., offset_height:offset_height+crop_shape[0], offset_width:offset_width+crop_shape[1]]
    elif data_format == 'channels_last':
        offset_height = (tf.shape(image)[-3] - crop_shape[0]) // 2
        offset_width = (tf.shape(image)[-2] - crop_shape[1]) // 2
        assert (offset_height >= 0) & (offset_width >= 0)
        image = image[..., offset_height:offset_height+crop_shape[0], offset_width:offset_width+crop_shape[1], :]
    else:
        raise ValueError(' ''data_format'' must be ''channels_first'' or ''channels_last''.')
    return image


def circular_pad(tf_input, pad, axis):
    """
    Perform circular padding.

    Inputs:
        tf_input: Tensor to pad values.
        pad: Number of values to pad in the beginning and at the end. Total amount of padding is 2xpad.
        axis: Axis to perform padding.
    
    Returns:
        tf_output: Circularly padded output.
    """
    shape_input = tf.shape(tf_input)
    shape_0 = tf.cast(tf.reduce_prod(shape_input[:axis]), dtype=tf.int32)
    shape_axis = shape_input[axis]
    tf_output = tf.reshape(tf_input, tf.stack((shape_0, shape_axis, -1)))

    tf_pre = tf_output[:, shape_axis - pad:, :]
    tf_post = tf_output[:, :pad, :]
    tf_output = tf.concat((tf_pre, tf_output, tf_post), axis=1)

    shape_out = tf.concat((shape_input[:axis],
                           [shape_axis + 2 * pad],
                           shape_input[axis + 1:]), axis=0)
    tf_output = tf.reshape(tf_output, shape_out)

    return tf_output


def add_quadratic_phase(image, phase_factor):
    """
    Add quadratic phase to image.

    Inputs:
        image: Complex valued image tensor of shape [..., H, W]
        phase_factor: Quadratic phase factor that determines the amount of phase to add.
            Added phase has the expression phase = `phase_factor` * pi * (y^2 + z^2)
    
    Returns:
        image: Complex valued image tensor with quadratic phase of shape [..., H, W]
    """

    h = tf.cast(tf.shape(image)[-2], tf.float32)
    w = tf.cast(tf.shape(image)[-1], tf.float32)
    hs = tf.range(h, dtype=tf.float32)/h*2 - 1
    ws = tf.range(w, dtype=tf.float32)/w*2 - 1
    hv, wv = tf.meshgrid(hs, ws, indexing='ij')
    quadratic_phase = phase_factor * math.pi * (hv**2 + wv**2)
    image = image * tf.exp(tf.complex(0., quadratic_phase))
    return image
