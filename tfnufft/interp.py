"""Interpolation functions.
"""

import tensorflow as tf
import numpy as np
from tfnufft import util

KERNELS = ['spline', 'kaiser_bessel']


def interpolate_noloop(input_tensor, coord, kernel='spline', width=2, param=1):
    """Interpolation from array to points specified by coordinates.
    Let :math:`x` be the input, :math:`y` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,
    .. math ::
        y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) x[i]
    There are two types of kernels: 'spline' and 'kaiser_bessel'.
    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.
    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.
    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.
    Args:
        input_tensor (tensor): Input tensor of shape [..., *img_shape[-ndim:]]..
        coord (tensor): Coordinate tensor of shape [..., ndim]
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {'spline', 'kaiser_bessel'}.
        param (float or tuple of floats): Kernel parameter.
    Returns:
        output_tensor (tensor): Output tensor.
    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = coord.shape[-1]
    batch_shape = tf.shape(input_tensor)[:-ndim]
    batch_size = tf.reduce_prod(batch_shape)
    image_shape = tf.shape(input_tensor)[-ndim:]

    pts_shape = tf.shape(coord)[:-1]
    npts = tf.reduce_prod(pts_shape)

    input_tensor = tf.reshape(input_tensor, tf.concat([[batch_size], image_shape], axis=0))
    coord = tf.reshape(coord, [npts, ndim])
    # output_tensor = tf.zeros([batch_size, npts], dtype=input_tensor.dtype)

    if tf.rank(param) == 0:
        param = tf.cast(tf.repeat(param, ndim), coord.dtype)
    else:
        param = tf.cast(param, coord.dtype)
    
    if tf.rank(width) == 0:
        width = tf.cast(tf.repeat(width, ndim), coord.dtype)
    else:
        width = tf.cast(width, coord.dtype)

    output_tensor = _interpolate_noloop(input_tensor, coord, width, param, kernel)

    # reshape to consistent dimensions
    output_tensor = tf.reshape(output_tensor, tf.concat([batch_shape, pts_shape], axis=0))

    return output_tensor


def interpolate_precalculated(input_tensor, ranges, w, pts_shape):
    """Interpolation from array to points specified by coordinates.
    This function uses precomputed kernel and index values.
    Precalculation must be performed using `get_kernel_values_indices`
    function.
    Let :math:`x` be the input, :math:`y` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,
    .. math ::
        y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) x[i]
    There are two types of kernels: 'spline' and 'kaiser_bessel'.
    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.
    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.
    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.
    Args:
        input_tensor (tensor): Input tensor of shape [..., *img_shape[-ndim:]]..
        ranges (tensor): Indices tensor compatible to use with gather_nd and scatter
            operations. Has shape [npts, ..., ndim]
        w (tensor): Precalculated kernel values of shape [npts, ...].
        pts_shape (tensor): 1d tensor representing the shape of `coord` tensor. 
    Returns:
        output_tensor (tensor): Output tensor.
    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = ranges.shape[-1]
    batch_shape = tf.shape(input_tensor)[:-ndim]
    batch_size = tf.reduce_prod(batch_shape)
    image_shape = tf.shape(input_tensor)[-ndim:]

    input_tensor = tf.reshape(input_tensor, tf.concat([[batch_size], image_shape], axis=0))
    # coord = tf.reshape(coord, [npts, ndim])

    output_tensor = _interpolate_precalculated(input_tensor, ranges, w)

    # reshape to consistent dimensions
    output_tensor = tf.reshape(output_tensor, tf.concat([batch_shape, pts_shape], axis=0))

    return output_tensor


def _interpolate_precalculated(input_tensor, ranges, w):
    """Interpolate from array to points using precomputed kernel and
    indices.
    Args:
        input_tensor (tensor): Input tensor of shape [batch_size, *img_shape[-ndim:]].
        ranges (tensor): Indices tensor compatible of shape [npts, ..., ndim].
        w (tensor): Precalculated kernel values of shape [npts, ...].
    Returns:
        output_tensor (tensor): Output tensor of shape [batch_size, npts].
    """

    ndim = ranges.shape[-1]
    w = tf.cast(w, dtype=input_tensor.dtype)

    input_tensor_selected_entries = tf.gather_nd(
        tf.transpose(input_tensor, perm=tf.roll(tf.range(ndim+1), shift=-1, axis=0)), 
        ranges
    ) # shape (npts, ..., batch_size)

    output_tensor = tf.reduce_sum(
        w[...,tf.newaxis] * input_tensor_selected_entries,
        axis=tf.range(1, ndim+1)
    ) # shape (npts, batch_size)
    output_tensor = tf.transpose(output_tensor, perm=[1,0]) # shape (batch_size, npts)

    return output_tensor


def interpolate(input_tensor, coord, kernel='spline', width=2, param=1):
    """Interpolation from array to points specified by coordinates.
    Let :math:`x` be the input, :math:`y` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,
    .. math ::
        y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) x[i]
    There are two types of kernels: 'spline' and 'kaiser_bessel'.
    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.
    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.
    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.
    Args:
        input_tensor (tensor): Input tensor of shape [..., *img_shape[-ndim:]].
        coord (tensor): Coordinate tensor of shape [..., ndim]
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {'spline', 'kaiser_bessel'}.
        param (float or tuple of floats): Kernel parameter.
    Returns:
        output_tensor (tensor): Output tensor.
    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = coord.shape[-1]
    batch_shape = tf.shape(input_tensor)[:-ndim]
    batch_size = tf.reduce_prod(batch_shape)
    image_shape = tf.shape(input_tensor)[-ndim:]

    pts_shape = tf.shape(coord)[:-1]
    npts = tf.reduce_prod(pts_shape)

    input_tensor = tf.reshape(input_tensor, tf.concat([[-1], image_shape], axis=0))
    coord = tf.reshape(coord, [npts, ndim])
    output_tensor = tf.zeros([batch_size, npts], dtype=input_tensor.dtype)

    if tf.rank(param) == 0:
        param = tf.cast(tf.repeat(param, ndim), coord.dtype)
    else:
        param = tf.cast(param, coord.dtype)
    
    if tf.rank(width) == 0:
        width = tf.cast(tf.repeat(width, ndim), coord.dtype)
    else:
        width = tf.cast(width, coord.dtype)

    output_tensor = _interpolate(output_tensor, input_tensor, coord, width, param, kernel)

    # reshape to consistent dimensions
    output_tensor = tf.reshape(output_tensor, tf.concat([batch_shape, pts_shape], axis=0))

    return output_tensor


def gridding_noloop(input_tensor, coord, img_shape, kernel="spline", width=2, param=1):
    """Gridding of points specified by coordinates to array.
    Let :math:`y` be the input, :math:`x` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,
    .. math ::
        x[i] = \sum_{j : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) y[j]
    There are two types of kernels: 'spline' and 'kaiser_bessel'.
    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.
    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.
    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.
    Args:
        input_tensor (tensor): Input tensor of shape [..., npts].
        coord (tensor): Coordinate tensor of shape [..., ndim]
        img_shape (tensor): Image shape to obtain after gridding.
            Including the batch dimension is optional, it will be ignored.
            Batch size will be inferred from `input_tensor`.
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {"spline", "kaiser_bessel"}.
        param (float or tuple of floats): Kernel parameter.
    Returns:
        output_tensor (tensor): Output tensor of shape [..., *img_shape[-ndim:]].
    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = coord.shape[-1]

    pts_shape = tf.shape(coord)[:-1]
    npts = tf.reduce_prod(pts_shape)

    batch_shape = tf.shape(input_tensor)[:-tf.rank(pts_shape)]
    batch_size = tf.reduce_prod(batch_shape)

    input_tensor = tf.reshape(input_tensor, [batch_size, npts])
    coord = tf.reshape(coord, [npts, ndim])

    output_tensor = tf.zeros(tf.concat([[batch_size], img_shape[-ndim:]], axis=0), dtype=input_tensor.dtype)

    if tf.rank(param) == 0:
        param = tf.cast(tf.repeat(param, ndim), coord.dtype)
    else:
        param = tf.cast(param, coord.dtype)

    if tf.rank(width) == 0:
        width = tf.cast(tf.repeat(width, ndim), coord.dtype)
    else:
        width = tf.cast(width, coord.dtype)

    output_tensor = _gridding_noloop(output_tensor, input_tensor, coord, width, param, kernel)

    # reshape to consistent dimensions
    output_tensor = tf.reshape(output_tensor, tf.concat([batch_shape, img_shape[-ndim:]], axis=0))

    return output_tensor


def gridding_precalculated(input_tensor, img_shape, ranges, w, pts_shape):
    """Gridding of points specified by coordinates to array.
    This function uses precomputed kernel and index values.
    Precalculation must be performed using `get_kernel_values_indices`
    function.
    Let :math:`y` be the input, :math:`x` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,
    .. math ::
        x[i] = \sum_{j : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) y[j]
    There are two types of kernels: 'spline' and 'kaiser_bessel'.
    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.
    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.
    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.
    Args:
        input_tensor (tensor): Input tensor of shape [..., npts].
        coord (tensor): Coordinate tensor of shape [..., ndim]
        img_shape (tensor): Image shape to obtain after gridding.
            Including the batch dimension is optional, it will be ignored.
            Batch size will be inferred from `input_tensor`.
        pts_shape (tensor): 1d tensor representing the shape of `coord` tensor. 
    Returns:
        output_tensor (tensor): Output tensor of shape [..., *img_shape[-ndim:]].
    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = ranges.shape[-1]

    # pts_shape = tf.shape(coord)[:-1]
    npts = tf.reduce_prod(pts_shape)

    batch_shape = tf.shape(input_tensor)[:-tf.rank(pts_shape)]
    batch_size = tf.reduce_prod(batch_shape)

    input_tensor = tf.reshape(input_tensor, [batch_size, npts])
    # coord = tf.reshape(coord, [npts, ndim])

    output_tensor = tf.zeros(tf.concat([[batch_size], img_shape[-ndim:]], axis=0), dtype=input_tensor.dtype)

    output_tensor = _gridding_precalculated(output_tensor, input_tensor, ranges, w)

    # reshape to consistent dimensions
    output_tensor = tf.reshape(output_tensor, tf.concat([batch_shape, img_shape[-ndim:]], axis=0))

    return output_tensor


def _gridding_precalculated(output_tensor, input_tensor, ranges, w):
    """Gridding of points to array using precomputed kernel and
    indices.
    Args:
        output_tensor (tensor): Output tensor of shape [batch_size, *img_shape[-ndim:]].
            Used as a placeholder to update the gridded values.
        input_tensor (tensor): Input tensor of shape [batch_size, npts].
        ranges (tensor): Indices tensor compatible of shape [npts, ..., ndim].
        w (tensor): Precalculated kernel values of shape [npts, ...].
    Returns:
        output_tensor (tensor): Output tensor of shape [batch_size, npts].
    """

    ndim = ranges.shape[-1]
    npts = tf.shape(ranges)[0]
    w = tf.cast(w, dtype=input_tensor.dtype)

    updates = w[..., tf.newaxis] *\
        tf.reshape(
            tf.transpose(input_tensor, perm=[1, 0]),
            tf.concat( [[npts], tf.ones(ndim, dtype=npts.dtype), [-1]], axis=0)
        )
    # updates shape: (npts, ..., batch_size)

    output_tensor = tf.tensor_scatter_nd_add(
        tf.transpose(output_tensor, perm=tf.roll(tf.range(ndim+1), shift=-1, axis=0)),
        ranges,
        updates
    )
    
    output_tensor = tf.transpose(output_tensor, perm=tf.roll(tf.range(ndim+1), shift=1, axis=0))

    return output_tensor

    
def gridding(input_tensor, coord, img_shape, kernel="spline", width=2, param=1):
    """Gridding of points specified by coordinates to array.
    Let :math:`y` be the input, :math:`x` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,
    .. math ::
        x[i] = \sum_{j : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) y[j]
    There are two types of kernels: 'spline' and 'kaiser_bessel'.
    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.
    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.
    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.
    Args:
        input_tensor (tensor): Input tensor of shape [..., npts].
        coord (tensor): Coordinate tensor of shape [..., ndim]
        img_shape (tensor): Image shape to obtain after gridding.
            Including the batch dimension is optional, it will be ignored.
            Batch size will be inferred from `input_tensor`.
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {"spline", "kaiser_bessel"}.
        param (float or tuple of floats): Kernel parameter.
    Returns:
        output_tensor (tensor): Output tensor of shape [..., *image_shape[-ndim:]].
    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """

    # We need the static shape for ndim here. It's not something dynamic.
    # In addition, it's fine in the @tf.function context as for each coord with
    # a different shape, a new function will be traced.
    ndim = coord.shape[-1]

    pts_shape = tf.shape(coord)[:-1]
    npts = tf.reduce_prod(pts_shape)

    batch_shape = tf.shape(input_tensor)[:-tf.rank(pts_shape)]
    batch_size = tf.reduce_prod(batch_shape)

    input_tensor = tf.reshape(input_tensor, [batch_size, npts])
    coord = tf.reshape(coord, [npts, ndim])

    output_tensor = tf.zeros(tf.concat([[batch_size], img_shape[-ndim:]], axis=0), dtype=input_tensor.dtype)

    if tf.rank(param) == 0:
        param = tf.cast(tf.repeat(param, ndim), coord.dtype)
    else:
        param = tf.cast(param, coord.dtype)

    if tf.rank(width) == 0:
        width = tf.cast(tf.repeat(width, ndim), coord.dtype)
    else:
        width = tf.cast(width, coord.dtype)

    output_tensor = _gridding(output_tensor, input_tensor, coord, width, param, kernel)

    # reshape to consistent dimensions
    output_tensor = tf.reshape(output_tensor, tf.concat([batch_shape, img_shape[-ndim:]], axis=0))

    return output_tensor


def get_kernel_values_indices(coord, os_shape, kernel, width, param):
    """Calculate kernel values for the `coord` values and specified array
    size `os_shape`. 
    Let :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,
    .. math ::
        K\left(\frac{i - c[j]}{W / 2}\right)
    There are two types of kernels: 'spline' and 'kaiser_bessel'.
    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.
    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.
    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.
    Args:
        coord (tensor): Coordinate tensor of shape [..., ndim]
        os_shape (tensor): Image shape in oversampled dimensions.
            Including the batch dimension is optional, it will be ignored.
        kernel (str): Interpolation kernel, {"spline", "kaiser_bessel"}.
        width (float or tuple of floats): Interpolation kernel full-width.
        param (float or tuple of floats): Kernel parameter.
    Returns:
        output_tensor (tensor): Output tensor of shape [..., *image_shape[-ndim:]].
    """

    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel

    ndim = coord.shape[-1]
    coord = tf.reshape(coord, [-1, ndim])
    npts = tf.shape(coord)[0]

    if tf.rank(param) == 0:
        param = tf.cast(tf.repeat(param, ndim), coord.dtype)
    else:
        param = tf.cast(param, coord.dtype)
    
    if tf.rank(width) == 0:
        width = tf.cast(tf.repeat(width, ndim), coord.dtype)
    else:
        width = tf.cast(width, coord.dtype)
    
    ns = os_shape[-ndim:] # oversampled image shape
    k0 = tf.math.ceil(coord - width/2)
    coord_indices = [k0[:,i,tf.newaxis] + tf.range(tf.math.floor(width[i])+1) for i in range(ndim)]
    coord_indices_circular = [tf.cast(coord_indices[i], ns[i].dtype) % ns[i] for i in range(ndim)]
    ranges = tf.vectorized_map(util.cartesian_product, coord_indices_circular) # (npts, ..., ndim)

    w = tf.ones([])
    for ii in range(ndim):
        kern = kernel((coord_indices[ii] - coord[:,ii,tf.newaxis]) / (width[ii] / 2), param[ii])
        w = w[...,tf.newaxis] * tf.reshape(kern, tf.concat([[npts],tf.ones([ii], dtype=npts.dtype),[-1]], axis=0))

    return w, ranges


def _spline_kernel_scalar(x, order):
    if abs(x) > 1:
        return 0

    if order == 0:
        return 1
    elif order == 1:
        return 1 - abs(x)
    elif order == 2:
        if abs(x) > 1 / 3:
            return 9 / 8 * (1 - abs(x))**2
        else:
            return 3 / 4 * (1 - 3 * x**2)


def _spline_kernel(x, order):
    if order == 0:
        out_tensor = tf.ones_like(x)
    elif order == 1:
        out_tensor = 1. - tf.abs(x)
    elif order == 2:
        out_tensor = tf.where(
            tf.abs(x) > 1/3,
            9 / 8 * (1 - tf.abs(x))**2,
            3 / 4 * (1 - 3 * x**2)
        )
    out_tensor = tf.where(
        tf.abs(x) > 1,
        0.,
        out_tensor
    )
    return out_tensor


def _kaiser_bessel_kernel_scalar(x, beta):
    if tf.abs(x) > 1:
        return 0

    x = beta * (1 - x**2)**0.5
    t = x / 3.75
    if x < 3.75:
        return 1 + 3.5156229 * t**2 + 3.0899424 * t**4 +\
            1.2067492 * t**6 + 0.2659732 * t**8 +\
            0.0360768 * t**10 + 0.0045813 * t**12
    else:
        return x**-0.5 * tf.exp(x) * (
            0.39894228 + 0.01328592 * t**-1 +
            0.00225319 * t**-2 - 0.00157565 * t**-3 +
            0.00916281 * t**-4 - 0.02057706 * t**-5 +
            0.02635537 * t**-6 - 0.01647633 * t**-7 +
            0.00392377 * t**-8)


def _kaiser_bessel_kernel_v1(x, beta):
    # if tf.abs(x) > 1:
        # return 0

    x = tf.convert_to_tensor(x)

    xx = tf.where(tf.abs(x)<=1, beta * (1 - x**2)**0.5, 3.75*tf.ones_like(x)) # avoid nans
    t = xx / 3.75

    out0 = tf.zeros_like(xx)

    out1 = 1 + 3.5156229 * t**2 + 3.0899424 * t**4 +\
            1.2067492 * t**6 + 0.2659732 * t**8 +\
            0.0360768 * t**10 + 0.0045813 * t**12
    
    out2 = xx**-0.5 * tf.exp(xx) * (
            0.39894228 + 0.01328592 * t**-1 +
            0.00225319 * t**-2 - 0.00157565 * t**-3 +
            0.00916281 * t**-4 - 0.02057706 * t**-5 +
            0.02635537 * t**-6 - 0.01647633 * t**-7 +
            0.00392377 * t**-8)
    
    output_tensor = tf.cast(tf.abs(x)>1, dtype=x.dtype) * out0 +\
                    tf.cast(tf.abs(x)<=1, dtype=x.dtype) * (
                        tf.cast(xx<3.75, dtype=x.dtype) * out1 +
                        tf.cast(xx>=3.75, dtype=x.dtype) * out2
                    )
    return output_tensor


def _kaiser_bessel_kernel(x, beta):
    beta = tf.cast(beta, x.dtype)
    # we need to use the repeated tf.where functions
    # to avoid NaN gradients when either branch generates
    # a nan, even though the result comes from the other branch.
    # this is documented in tf.where documentation as well as
    # in https://github.com/tensorflow/tensorflow/issues/38349.
    safe_x = tf.where(tf.abs(x) > 1., 0., x)
    kern = tf.math.bessel_i0(beta * tf.sqrt(1 - safe_x ** 2))
    return tf.where(tf.abs(x) > 1., 0., kern)


def _griddingn(output_tensor, input_tensor, coord, width, param, kernel):
    batch_size = tf.shape(output_tensor)[0]
    npts = tf.shape(coord)[0]
    # important (for tf.graph context): we need the static shape here.
    # Otherwise the dynamic shape (tf.shape(...)) would be unknown in tf.function
    # context and it would fail in graph mode. As for each coord tensor shape,
    # a new concrete function would be generated. Therefore, it is completely
    # fine to use the static shape here.
    ndim = coord.shape[-1]
    ns = tf.shape(output_tensor)[-ndim:]

    for i in range(npts):
        ks = coord[i, -ndim:]

        k0 = tf.math.ceil(ks - width/2)
        k1 = tf.math.floor(ks + width/2)
        k0 = tf.reshape(k0, [ndim])
        k1 = tf.reshape(k1, [ndim])

        coord_indices = [tf.range(xx,yy) for xx,yy in zip(tf.unstack(k0), tf.unstack(k1+1))]
        coord_indices_circular = [tf.cast(xx,yy.dtype) % yy for xx,yy in zip(coord_indices, tf.unstack(ns))]
        w = tf.ones([])
        for ii in range(len(coord_indices)):
            kern = kernel((coord_indices[ii] - ks[ii]) / (width[ii] / 2), param[ii])
            w = w[...,tf.newaxis] * tf.reshape(kern, tf.concat([tf.ones([ii],dtype=tf.int32),[-1]], axis=0))
        w = tf.cast(w, dtype=input_tensor.dtype)

        # TF way to do output_tensor[:, *coord_indices_circular] += w[None] * input_tensor[:,i]
        output_tensor = tf.tensor_scatter_nd_add(
            tf.transpose(output_tensor, perm=tf.roll(tf.range(ndim+1), shift=-1, axis=0)),
            util.cartesian_product(coord_indices_circular),
            w[...,tf.newaxis] * input_tensor[:,i]
        )
        output_tensor = tf.transpose(output_tensor, perm=tf.roll(tf.range(ndim+1), shift=1, axis=0))

    return output_tensor


def _griddingn_noloop(output_tensor, input_tensor, coord, width, param, kernel):

    batch_size = tf.shape(input_tensor)[0]
    npts = tf.shape(coord)[0]
    # important (for tf.graph context): we need the static shape here.
    # Otherwise the dynamic shape (tf.shape(...)) would be unknown in tf.function
    # context and it would fail in graph mode. As for each coord tensor shape,
    # a new concrete function would be generated. Therefore, it is completely
    # fine to use the static shape here.
    ndim = coord.shape[-1]
    ns = tf.shape(output_tensor)[-ndim:]

    k0 = tf.math.ceil(coord - width/2)
    # k1 = k0 + tf.math.floor(width)

    coord_indices = [k0[:,i,tf.newaxis] + tf.range(tf.math.floor(width[i])+1) for i in range(ndim)]
    coord_indices_circular = [tf.cast(coord_indices[i], ns[i].dtype) % ns[i] for i in range(ndim)]
    ranges = tf.vectorized_map(util.cartesian_product, coord_indices_circular) # (npts, ..., ndim)
    # ranges = tf.map_fn(util.cartesian_product, coord_indices_circular, fn_output_signature=ns.dtype) # (npts, ..., ndim)

    w = tf.ones([])
    for ii in range(ndim):
        kern = kernel((coord_indices[ii] - coord[:,ii,tf.newaxis]) / (width[ii] / 2), param[ii])
        w = w[...,tf.newaxis] * tf.reshape(kern, tf.concat([[npts],tf.ones([ii], dtype=npts.dtype),[-1]], axis=0))
    # w now has shape (npts, ...)
    w = tf.cast(w, dtype=input_tensor.dtype)

    updates = w[..., tf.newaxis] *\
        tf.reshape(
            tf.transpose(input_tensor, perm=[1, 0]),
            tf.concat( [[npts], tf.ones(ndim, dtype=npts.dtype), [-1]], axis=0)
        )
    # updates shape: (npts, ..., batch_size)

    output_tensor = tf.tensor_scatter_nd_add(
        tf.transpose(output_tensor, perm=tf.roll(tf.range(ndim+1), shift=-1, axis=0)),
        ranges,
        updates
    )
    
    output_tensor = tf.transpose(output_tensor, perm=tf.roll(tf.range(ndim+1), shift=1, axis=0))

    return output_tensor


def _interpolaten(output_tensor, input_tensor, coord, width, param, kernel):

    batch_size = tf.shape(output_tensor)[0]
    npts = tf.shape(coord)[0]
    # important (for tf.graph context): we need the static shape here.
    # Otherwise the dynamic shape (tf.shape(...)) would be unknown in tf.function
    # context and it would fail in graph mode. As for each coord tensor shape,
    # a new concrete function would be generated. Therefore, it is completely
    # fine to use the static shape here.
    ndim = coord.shape[-1]
    ns = tf.shape(input_tensor)[-ndim:]

    for i in range(npts):
        ks = coord[i, -ndim:]

        k0 = tf.math.ceil(ks - width/2)
        k1 = tf.math.floor(ks + width/2)
        k0 = tf.reshape(k0, [ndim])
        k1 = tf.reshape(k1, [ndim])

        coord_indices = [tf.range(xx,yy) for xx,yy in zip(tf.unstack(k0), tf.unstack(k1+1))]
        coord_indices_circular = [tf.cast(xx,yy.dtype) % yy for xx,yy in zip(coord_indices, tf.unstack(ns))]
        w = tf.ones([])
        for ii in range(len(coord_indices)):
            kern = kernel((coord_indices[ii] - ks[ii]) / (width[ii] / 2), param[ii])
            w = w[...,tf.newaxis] * tf.reshape(kern, tf.concat([tf.ones([ii],dtype=tf.int32),[-1]], axis=0))
        w = tf.cast(w, dtype=input_tensor.dtype)
        
        ranges = util.cartesian_product(coord_indices_circular)
        input_tensor_selected_entries = tf.gather_nd(
            tf.transpose(input_tensor, perm=tf.roll(tf.range(ndim+1), shift=-1, axis=0)),
            ranges
        ) # shape (..., batch_size)

        # TF way to do output_tensor[:,i] += dot_product(w,input_tensor[:,*coord_indices_circular])
        output_tensor = tf.tensor_scatter_nd_add(
            tf.transpose(output_tensor, perm=[1,0]),
            [[i]],
            tf.tensordot(w[tf.newaxis], input_tensor_selected_entries, axes=ndim) # shape (1, batch_size)
        )
        output_tensor = tf.transpose(output_tensor, perm=[1,0])

    return output_tensor


def _interpolaten_noloop(input_tensor, coord, width, param, kernel):

    batch_size = tf.shape(input_tensor)[0]
    npts = tf.shape(coord)[0]
    # important (for tf.graph context): we need the static shape here.
    # Otherwise the dynamic shape (tf.shape(...)) would be unknown in tf.function
    # context and it would fail in graph mode. As for each coord tensor shape,
    # a new concrete function would be generated. Therefore, it is completely
    # fine to use the static shape here.
    ndim = coord.shape[-1]
    ns = tf.shape(input_tensor)[-ndim:]

    k0 = tf.math.ceil(coord - width/2)
    # k1 = k0 + tf.math.floor(width)

    coord_indices = [k0[:,i,tf.newaxis] + tf.range(tf.math.floor(width[i])+1) for i in range(ndim)]
    coord_indices_circular = [tf.cast(coord_indices[i], ns[i].dtype) % ns[i] for i in range(ndim)]
    ranges = tf.vectorized_map(util.cartesian_product, coord_indices_circular) # (npts, ..., ndim)
    # ranges = tf.map_fn(util.cartesian_product, coord_indices_circular, fn_output_signature=ns.dtype) # (npts, ..., ndim)

    w = tf.ones([])
    for ii in range(ndim):
        kern = kernel((coord_indices[ii] - coord[:,ii,tf.newaxis]) / (width[ii] / 2), param[ii])
        w = w[...,tf.newaxis] * tf.reshape(kern, tf.concat([[npts],tf.ones([ii], dtype=npts.dtype),[-1]], axis=0))
    # w now has shape (npts, ...)
    w = tf.cast(w, dtype=input_tensor.dtype)

    input_tensor_selected_entries = tf.gather_nd(
        tf.transpose(input_tensor, perm=tf.roll(tf.range(ndim+1), shift=-1, axis=0)), 
        ranges
    ) # shape (npts, ..., batch_size)

    output_tensor = tf.reduce_sum(
        w[...,tf.newaxis] * input_tensor_selected_entries,
        axis=tf.range(1, ndim+1)
    ) # shape (npts, batch_size)
    output_tensor = tf.transpose(output_tensor, perm=[1,0]) # shape (batch_size, npts)

    return output_tensor


def _interpolate(output_tensor, input_tensor, coord, width, param, kernel):
    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel
    
    return _interpolaten(output_tensor, input_tensor, coord, width, param, kernel)


def _interpolate_noloop(input_tensor, coord, width, param, kernel):
    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel
    
    return _interpolaten_noloop(input_tensor, coord, width, param, kernel)


def _gridding(output_tensor, input_tensor, coord, width, param, kernel):
    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel
    
    return _griddingn(output_tensor, input_tensor, coord, width, param, kernel)


def _gridding_noloop(output_tensor, input_tensor, coord, width, param, kernel):
    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel
    
    return _griddingn_noloop(output_tensor, input_tensor, coord, width, param, kernel)
    