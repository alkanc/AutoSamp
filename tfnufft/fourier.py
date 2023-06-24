"""FFT and non-uniform FFT (NUFFT) functions.
"""
import math

import tensorflow as tf

from tfnufft import util
from tfnufft import interp


def fft(input_tensor, fftdim, oshape=None, center=True, norm='ortho'):
    """FFT function that supports centering.
    Args:
        input_tensor (tensor): input tensor.
        fftdim (int): number of dimensions to apply the fft. 
            For example, if fftdim=2, 2d fft is taken along innermost dims.
        oshape (None or array of ints): output shape.
        center (bool): centered fft or not.
        norm (``"ortho"`` or ``"forward"`` or ``"backward"``): Keyword to specify the normalization mode.
    Returns:
        output_tensor: FFT result tensor of dimension oshape.
    """
    input_tensor = tf.cast(input_tensor, dtype=tf.complex64)

    if center:
        output_tensor = _fftc(input_tensor, fftdim, oshape=oshape, norm=norm)
    else:
        output_tensor = _fftn(input_tensor, fftdim, oshape=oshape, norm=norm)

    return output_tensor


def ifft(input_tensor, fftdim, oshape=None, center=True, norm='ortho'):
    """IFFT function that supports centering.
    Args:
        input_tensor (tensor): input tensor.
        fftdim (int): number of dimensions to apply the ifft. 
            For example, if fftdim=2, 2d ifft is taken along innermost dims.
        oshape (None or array of ints): output shape.
        norm (``"ortho"`` or ``"forward"`` or ``"backward"``): Keyword to specify the normalization mode.
    Returns:
        output_tensor: iFFT result tensor of dimension oshape.
    """
    input_tensor = tf.cast(input_tensor, dtype=tf.complex64)

    if center:
        output_tensor = _ifftc(input_tensor, fftdim, oshape=oshape, norm=norm)
    else:
        output_tensor = _ifftn(input_tensor, fftdim, oshape=oshape, norm=norm)

    return output_tensor


def _fftc(input_tensor, fftdim, oshape=None, norm='ortho'):
    if oshape is None:
        oshape = tf.shape(input_tensor)
    
    if fftdim > 3 or fftdim < 1:
        raise ValueError('Only 1d, 2d and 3d fft is supported.')
    axes = tf.rank(input_tensor) + tf.range(start=-1, limit=-fftdim-1, delta=-1)

    tmp = util.resize(input_tensor, oshape)
    tmp = tf.signal.ifftshift(tmp, axes=axes)
    tmp = _fftn(tmp, fftdim, norm=norm)
    output_tensor = tf.signal.fftshift(tmp, axes=axes)
    return output_tensor


def _ifftc(input_tensor, fftdim, oshape=None, norm='ortho'):
    if oshape is None:
        oshape = tf.shape(input_tensor)
    
    if fftdim > 3 or fftdim < 1:
        raise ValueError('Only 1d, 2d and 3d fft is supported.')
    axes = tf.rank(input_tensor) + tf.range(start=-1, limit=-fftdim-1, delta=-1)

    tmp = util.resize(input_tensor, oshape)
    tmp = tf.signal.ifftshift(tmp, axes=axes)
    tmp = _ifftn(tmp, fftdim, norm=norm)
    output_tensor = tf.signal.fftshift(tmp, axes=axes)
    return output_tensor


def _fftn(input_tensor, fftdim, oshape=None, norm='ortho'):
    if oshape is not None:
        input_tensor = util.resize(input_tensor, oshape)

    if fftdim > 3 or fftdim < 1:
        raise ValueError('Only 1d, 2d and 3d fft is supported.')
    
    # calculate normalization
    # note that TF FFT operations work in 'backward' normalization mode
    axes = tf.rank(input_tensor) + tf.range(start=-1, limit=-fftdim-1, delta=-1)
    num_elements = tf.reduce_prod(tf.gather(tf.shape(input_tensor), axes))
    num_elements = tf.cast(num_elements, tf.float32)
    if norm == 'ortho':
        scale = tf.math.divide_no_nan(1., tf.sqrt(num_elements))
    elif norm == 'forward':
        scale = tf.math.divide_no_nan(1., num_elements)
    elif norm == 'backward':
        scale = tf.constant(1.0)
    else:
        raise ValueError('Norm `%s` is not supported')
    
    # use the correct fft function
    if fftdim == 1:
        fftfun = tf.signal.fft
    elif fftdim == 2:
        fftfun = tf.signal.fft2d
    elif fftdim == 3:
        fftfun = tf.signal.fft3d

    output_tensor = tf.cast(scale, input_tensor.dtype) * fftfun(input_tensor)
    return output_tensor


def _ifftn(input_tensor, fftdim, oshape=None, norm='ortho'):
    if oshape is not None:
        input_tensor = util.resize(input_tensor, oshape)
    
    if fftdim > 3 or fftdim < 1:
        raise ValueError('Only 1d, 2d and 3d fft is supported.')
    
    # calculate normalization
    # note that TF FFT operations work in 'backward' normalization mode
    axes = tf.rank(input_tensor) + tf.range(start=-1, limit=-fftdim-1, delta=-1)
    num_elements = tf.reduce_prod(tf.gather(tf.shape(input_tensor), axes))
    num_elements = tf.cast(num_elements, tf.float32)
    if norm == 'ortho':
        scale = tf.sqrt(num_elements)
    elif norm == 'forward':
        scale = num_elements
    elif norm == 'backward':
        scale = tf.constant(1.0)
    else:
        raise ValueError('Norm `%s` is not supported')
    
    # use the correct fft function
    if fftdim == 1:
        fftfun = tf.signal.ifft
    elif fftdim == 2:
        fftfun = tf.signal.ifft2d
    elif fftdim == 3:
        fftfun = tf.signal.ifft3d

    output_tensor = tf.cast(scale, input_tensor.dtype) * fftfun(input_tensor)
    return output_tensor


def _apodize(input_tensor, ndim, oversamp, width, beta):
    output_tensor = input_tensor
    for a in range(-ndim, 0):
        i_int = tf.shape(output_tensor)[a]
        i = tf.cast(i_int, tf.float32)
        os_i = tf.math.ceil(oversamp * i)
        idx = tf.range(i, dtype=tf.float32)

        # calculate apodization
        sqrt_arg = beta**2 - (math.pi * width * (idx - i // 2) / os_i)**2
        # convert to complex in case `sqrt_arg` is negative
        sqrt_arg = tf.cast(sqrt_arg, tf.complex64)
        apod = tf.math.sqrt(sqrt_arg)
        apod = tf.math.divide_no_nan(apod, tf.math.sinh(apod))
        apod = tf.reshape(apod, tf.concat([i_int[tf.newaxis], tf.repeat(1, -a-1)], axis=0))
        output_tensor = output_tensor * tf.cast(apod, output_tensor.dtype)

    return output_tensor


def _get_oversamp_shape(shape, ndim, oversamp):
    return tf.concat([shape[:-ndim],
                      tf.cast(tf.math.ceil(tf.cast(shape[-ndim:], tf.float32) * oversamp), tf.int32)], axis=0)


def _scale_coord(coord, shape, oversamp):
    ndim = tf.shape(coord)[-1]
    scale = tf.math.divide_no_nan(tf.math.ceil(tf.cast(shape[-ndim:], tf.float32) * oversamp), 
                                  tf.cast(shape[-ndim:], tf.float32))
    shift = tf.math.ceil(tf.cast(shape[-ndim:], tf.float32) * oversamp) // 2
    output_tensor = coord * scale + shift
    return output_tensor


# @tf.function
def get_nufft_obj(coord, oshape, oversamp=1.25, width=4., kernel='kaiser_bessel'):
    """Create nufft object by precomputing kernel and index values.  
    Args:
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimensions to apply the nufft.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
             (n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
    Returns:
        nufft_obj (dict): dict containing precomputed kernel and index
            tensors for computing nufft and nufft_adjoint.
    """

    oversamp = tf.convert_to_tensor(oversamp, dtype=tf.float32)
    width = tf.convert_to_tensor(width, dtype=tf.float32)
    coord = tf.convert_to_tensor(coord, dtype=tf.float32)
    oshape = tf.convert_to_tensor(oshape)
    ndim = coord.shape[-1]
    pts_shape = tf.shape(coord)[:-1]

    nufft_obj = dict()
    beta = math.pi * tf.math.sqrt(((width / oversamp) * (oversamp - 0.5))**2 - 0.8)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)
    coord = _scale_coord(coord, oshape, oversamp)

    w, ranges = interp.get_kernel_values_indices(
        coord, os_shape, kernel=kernel, width=width, param=beta)

    nufft_obj['kernel'] = kernel
    nufft_obj['oversamp'] = oversamp
    nufft_obj['width'] = width
    nufft_obj['beta'] = beta
    nufft_obj['oshape'] = oshape
    nufft_obj['os_shape'] = os_shape
    nufft_obj['w'] = w
    nufft_obj['ranges'] = ranges
    nufft_obj['pts_shape'] = pts_shape

    return nufft_obj


def nufft_from_obj(nufft_obj):
    """Non-uniform Fast Fourier Transform from precomputed kernel.
    Args:
        nufft_obj (dict): nuFFT object calculated using `get_nufft_obj`.
    Returns:
        _nufft: nuFFT function that uses precomputed values specified in
            `nufft_obj`.
    """
    # @tf.function
    def _nufft(input_tensor):
        """Non-uniform Fast Fourier Transform.
        Args:
            input_tensor (array): input signal domain tensor of shape
                (..., n_{ndim - 1}, ..., n_1, n_0),
                where ndim is specified by coord.shape[-1]. The nufft
                is applied on the last ndim axes, and looped over
                the remaining axes.
        Returns:
            output_tensor: Fourier domain data of shape
                input.shape[:-ndim] + coord.shape[:-1].
        References:
            Fessler, J. A., & Sutton, B. P. (2003).
            Nonuniform fast Fourier transforms using min-max interpolation
            IEEE Transactions on Signal Processing, 51(2), 560-574.
            Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
            Rapid gridding reconstruction with a minimal oversampling ratio.
            IEEE transactions on medical imaging, 24(6), 799-808.
        """
        input_tensor = tf.convert_to_tensor(input_tensor)

        kernel = nufft_obj['kernel']
        oversamp = nufft_obj['oversamp']
        width = nufft_obj['width']
        beta = nufft_obj['beta']
        oshape = nufft_obj['oshape']
        os_shape = nufft_obj['os_shape']
        w = nufft_obj['w']
        ranges = nufft_obj['ranges']
        pts_shape = nufft_obj['pts_shape']

        ndim = ranges.shape[-1]

        # need to add the batch dims to oshape and os_shape
        batch_shape = tf.shape(input_tensor)[:-ndim]
        oshape = tf.concat([batch_shape, oshape[-ndim:]], axis=0)
        os_shape = tf.concat([batch_shape, os_shape[-ndim:]], axis=0)

        # Apodize
        output_tensor = _apodize(input_tensor, ndim, oversamp, width, beta)

        # Zero-pad
        output_tensor = output_tensor / tf.math.sqrt(tf.cast(tf.reduce_prod(tf.shape(input_tensor)[-ndim:]), output_tensor.dtype))
        output_tensor = util.resize(output_tensor, os_shape)

        # FFT
        output_tensor = fft(output_tensor, fftdim=ndim, norm='backward')

        # Interpolate
        output_tensor = interp.interpolate_precalculated(
            output_tensor, ranges=ranges, w=w, pts_shape=pts_shape
        )
        output_tensor = output_tensor / tf.cast(width**ndim, output_tensor.dtype)
        return output_tensor
    return _nufft


def nufft_adjoint_from_obj(nufft_obj):
    """Adjoint non-uniform Fast Fourier Transform from precomputed kernel.
    Args:
        nufft_obj (dict): nuFFT object calculated using `get_nufft_obj`.
    Returns:
        _nufft_adjoint: nuFFT function that uses precomputed values 
            specified in `nufft_obj`.
    """
    # @tf.function
    def _nufft_adjoint(input_tensor):
        input_tensor = tf.convert_to_tensor(input_tensor)

        kernel = nufft_obj['kernel']
        oversamp = nufft_obj['oversamp']
        width = nufft_obj['width']
        beta = nufft_obj['beta']
        oshape = nufft_obj['oshape']
        os_shape = nufft_obj['os_shape']
        w = nufft_obj['w']
        ranges = nufft_obj['ranges']
        pts_shape = nufft_obj['pts_shape']

        ndim = ranges.shape[-1]

        # need to add the batch dims to oshape and os_shape
        batch_shape = tf.shape(input_tensor)[:-tf.rank(pts_shape)]
        oshape = tf.concat([batch_shape, oshape[-ndim:]], axis=0)
        os_shape = tf.concat([batch_shape, os_shape[-ndim:]], axis=0)

        # Gridding
        output_tensor = interp.gridding_precalculated(
            input_tensor, os_shape, ranges=ranges, w=w, pts_shape=pts_shape
        )
        output_tensor = output_tensor / tf.cast(width**ndim, output_tensor.dtype)

        # IFFT
        output_tensor = ifft(output_tensor, fftdim=ndim, norm='backward')

        # Crop
        output_tensor = util.resize(output_tensor, oshape)
        output_tensor = output_tensor * \
            tf.cast(tf.reduce_prod(os_shape[-ndim:]), output_tensor.dtype) / \
            tf.math.sqrt(tf.cast(tf.reduce_prod(oshape[-ndim:]), output_tensor.dtype))

        # Apodize
        output_tensor = _apodize(output_tensor, ndim, oversamp, width, beta)

        return output_tensor

    return _nufft_adjoint


@tf.function
def nufft(input_tensor, coord, oversamp=1.25, width=4.):
    """Non-uniform Fast Fourier Transform.
    Args:
        input_tensor (array): input signal domain tensor of shape
            (..., n_{ndim - 1}, ..., n_1, n_0),
            where ndim is specified by coord.shape[-1]. The nufft
            is applied on the last ndim axes, and looped over
            the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimensions to apply the nufft.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
    Returns:
        output_tensor: Fourier domain data of shape
            input.shape[:-ndim] + coord.shape[:-1].
    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.
    """

    input_tensor = tf.convert_to_tensor(input_tensor)
    coord = tf.convert_to_tensor(coord, dtype=tf.float32)
    oversamp = tf.convert_to_tensor(oversamp, dtype=tf.float32)
    width = tf.convert_to_tensor(width, dtype=tf.float32)

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = coord.shape[-1]
    beta = math.pi * tf.math.sqrt(((width / oversamp) * (oversamp - 0.5))**2 - 0.8)
    os_shape = _get_oversamp_shape(tf.shape(input_tensor), ndim, oversamp)

    # Apodize
    output_tensor = _apodize(input_tensor, ndim, oversamp, width, beta)

    # Zero-pad
    output_tensor = output_tensor / tf.math.sqrt(tf.cast(tf.reduce_prod(tf.shape(input_tensor)[-ndim:]), output_tensor.dtype))
    output_tensor = util.resize(output_tensor, os_shape)

    # FFT
    output_tensor = fft(output_tensor, fftdim=ndim, norm='backward')

    # Interpolate
    coord = _scale_coord(coord, tf.shape(input_tensor), oversamp)
    output_tensor = interp.interpolate_noloop(
        output_tensor, coord, kernel='kaiser_bessel', width=width, param=beta
    )
    output_tensor = output_tensor / tf.cast(width**ndim, output_tensor.dtype)

    return output_tensor


@tf.function
def nufft_adjoint(input_tensor, coord, oshape, oversamp=1.25, width=4.):
    """Adjoint non-uniform Fast Fourier Transform.
    Args:
        input_tensor (tensor): input Fourier domain tensor of shape
            (...) + coord.shape[:-1]. That is, the last dimensions
            of input must match the first dimensions of coord.
            The nufft_adjoint is applied on the last coord.ndim - 1 axes,
            and looped over the remaining axes.
        coord (tensor): Fourier domain coordinate tensor of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
             (n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
    Returns:
        output_tensor: signal domain tensor with shape specified by oshape.
    """

    input_tensor = tf.convert_to_tensor(input_tensor)
    coord = tf.convert_to_tensor(coord)
    oversamp = tf.convert_to_tensor(oversamp, dtype=tf.float32)
    width = tf.convert_to_tensor(width, dtype=tf.float32)

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = coord.shape[-1]
    beta = math.pi * tf.math.sqrt(((width / oversamp) * (oversamp - 0.5))**2 - 0.8)

    pts_axes_len = tf.rank(coord) - 1
    batch_shape = tf.shape(input_tensor)[:-pts_axes_len]
    oshape = tf.concat([batch_shape, oshape], axis=0)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    output_tensor = interp.gridding_noloop(input_tensor, coord, os_shape,
                                    kernel='kaiser_bessel', width=width, param=beta)
    output_tensor = output_tensor / tf.cast(width**ndim, output_tensor.dtype)

    # IFFT
    output_tensor = ifft(output_tensor, fftdim=ndim, norm='backward')

    # Crop
    output_tensor = util.resize(output_tensor, oshape)
    output_tensor = output_tensor * \
        tf.cast(tf.reduce_prod(os_shape[-ndim:]), output_tensor.dtype) / \
        tf.math.sqrt(tf.cast(tf.reduce_prod(oshape[-ndim:]), output_tensor.dtype))

    # Apodize
    output_tensor = _apodize(output_tensor, ndim, oversamp, width, beta)

    return output_tensor

