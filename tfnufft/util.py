"""Utility functions.
"""

import tensorflow as tf

def _expand_shapes(*shapes):
    max_ndim = tf.math.reduce_max([tf.size(sh) for sh in shapes])
    shapes_exp = [tf.pad(sh, paddings=[[max_ndim-tf.size(sh), 0]], constant_values=1)
                  for sh in shapes]
    return shapes_exp

def resize_v1(input_tensor, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.
    Args:
        input_tensor (tensor): Input tensor.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.
    Returns:
        output_tensor: Zero-padded or cropped result.
    """

    ishape1, oshape1 = _expand_shapes(tf.shape(input_tensor), oshape)

    # if tf.reduce_all(ishape1 == oshape1):
    #     return tf.reshape(input_tensor, oshape)

    if ishift is None:
        # ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]
        ishift = [tf.math.maximum(tf.math.floordiv(i,2) - tf.math.floordiv(o,2), 0)
                  for i, o in zip(tf.unstack(ishape1), tf.unstack(oshape1))]

    if oshift is None:
        # oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]
        oshift = [tf.math.maximum(tf.math.floordiv(o,2) - tf.math.floordiv(i,2), 0)
                  for i, o in zip(tf.unstack(ishape1), tf.unstack(oshape1))]

    copy_shape = [tf.math.minimum(i - si, o - so)
                  for i, si, o, so in zip(tf.unstack(ishape1), ishift, tf.unstack(oshape1), oshift)]
    islice = []

    islice = [slice(si, si + c) for si, c in zip(ishift, copy_shape)]
    oslice = [slice(so, so + c) for so, c in zip(oshift, copy_shape)]

    # using Variable instead of a Tensor for tensor assignment
    # bad solution, but tensorflow's tensor assignment logic is not great
    # note that creating a new variable is not allowed in tf.function context
    # therefore this code will not work in graph mode
    # reference: https://github.com/tensorflow/tensorflow/issues/33131
    output_tensor = tf.Variable(tf.zeros(oshape1, dtype=input_tensor.dtype))
    input_tensor = tf.reshape(input_tensor, ishape1)
    _ = output_tensor[oslice].assign(input_tensor[islice])

    return tf.convert_to_tensor(output_tensor)


def resize(input_tensor, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.
    Args:
        input_tensor (tensor): Input tensor.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.
    Returns:
        output_tensor: Zero-padded or cropped result.
    """

    ishape1, oshape1 = _expand_shapes(tf.shape(input_tensor), oshape)

    # if tf.reduce_all(ishape1 == oshape1):
    #     return tf.reshape(input_tensor, oshape)

    if ishift is None:
        # ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]
        ishift = [tf.math.maximum(tf.math.floordiv(i,2) - tf.math.floordiv(o,2), 0)
                  for i, o in zip(tf.unstack(ishape1), tf.unstack(oshape1))]

    if oshift is None:
        # oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]
        oshift = [tf.math.maximum(tf.math.floordiv(o,2) - tf.math.floordiv(i,2), 0)
                  for i, o in zip(tf.unstack(ishape1), tf.unstack(oshape1))]

    copy_shape = [tf.math.minimum(i - si, o - so)
                  for i, si, o, so in zip(tf.unstack(ishape1), ishift, tf.unstack(oshape1), oshift)]

    # islice = [slice(si, si + c) for si, c in zip(ishift, copy_shape)]
    # oslice = [slice(so, so + c) for so, c in zip(oshift, copy_shape)]

    islice = [tf.range(si, si + c) for si, c in zip(ishift, copy_shape)]
    oslice = [tf.range(so, so + c) for so, c in zip(oshift, copy_shape)]

    # can use Variable instead of a Tensor for tensor assignment
    # bad solution, but tensorflow's tensor assignment logic is not great
    # note that creating a new variable is not allowed in tf.function context
    # therefore the commented code will not work in graph mode
    # reference: https://github.com/tensorflow/tensorflow/issues/33131
    # output_tensor = tf.Variable(tf.zeros(oshape1, dtype=input_tensor.dtype))
    # input_tensor = tf.reshape(input_tensor, ishape1)
    # _ = output_tensor[oslice].assign(input_tensor[islice])

    # The alternative is using scatter_nd and gather_nd
    iranges = cartesian_product(islice)
    oranges = cartesian_product(oslice)
    output_tensor = tf.scatter_nd(
        oranges,
        tf.gather_nd(input_tensor, iranges),
        shape=oshape1
    )

    return tf.convert_to_tensor(output_tensor)


def cartesian_product(args, indexing='ij'):
    """Calculate cartesian product between given vectors.
    
    Args:
        args: List (or tuple) of tensors to take the cartesian product.
        indexing (str): 'ij' (matrix) or 'xy' (cartesian) indexing convention, see tf.meshgrid.
    Returns:
        cart_prod: Cartesian product of tensors given in args.
            Has shape [..., N] where N is the number of tensors.
    """

    mesh_indices = tf.meshgrid(*args, indexing=indexing)
    cart_prod = tf.stack(mesh_indices, axis=-1)
    return cart_prod

