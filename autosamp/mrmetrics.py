from absl import app
from absl import flags

import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import autosamp.mrutils as mrutils


class MeanSquaredComplexError(tf.keras.metrics.Metric):
    '''
    Metric that calculates mean squared error given true and predicted complex valued tensors.
    '''
    def __init__(self, name='mean_squared_complex_error', **kwargs):
        super(MeanSquaredComplexError, self).__init__(name=name, **kwargs)
        self.total_msce = self.add_weight(name='total_mean_squared_complex_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # dimensions except batch dim must be reduced
        dims_to_reduce = tf.range(1, tf.rank(y_true))

        squared_error = tf.square(tf.abs(y_true - y_pred))
        msce = tf.reduce_mean(squared_error, axis=dims_to_reduce)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            msce = tf.multiply(msce, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(msce), tf.float32))
        self.total_msce.assign_add(tf.reduce_sum(msce))

    def result(self):
        return tf.math.divide_no_nan(self.total_msce, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_msce.assign(0.)
        self.count.assign(0.)


class MeanAbsoluteComplexError(tf.keras.metrics.Metric):
    '''
    Metric that calculates mean absolute given true and predicted complex valued tensors.
    '''
    def __init__(self, name='mean_absolute_complex_error', **kwargs):
        super(MeanAbsoluteComplexError, self).__init__(name=name, **kwargs)
        self.total_mae = self.add_weight(name='total_mean_absolute_complex_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # dimensions except batch dim must be reduced
        dims_to_reduce = tf.range(1, tf.rank(y_true))

        absolute_error = tf.abs(y_true - y_pred)
        mae = tf.reduce_mean(absolute_error, axis=dims_to_reduce)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            mae = tf.multiply(mae, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(mae), tf.float32))
        self.total_mae.assign_add(tf.reduce_sum(mae))

    def result(self):
        return tf.math.divide_no_nan(self.total_mae, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_mae.assign(0.)
        self.count.assign(0.)


class NormalizedRootMeanSquaredComplexError(tf.keras.metrics.Metric):
    '''
    Metric that calculates normalized root mean squared error given true and predicted complex valued tensors.
    '''
    def __init__(self, name='normalized_root_mean_squared_complex_error', **kwargs):
        super(NormalizedRootMeanSquaredComplexError, self).__init__(name=name, **kwargs)
        self.total_nrmsce = self.add_weight(name='total_normalized_root_mean_squared_complex_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # dimensions except batch dim must be reduced
        dims_to_reduce = tf.range(1, tf.rank(y_true))

        squared_error = tf.square(tf.abs(y_true - y_pred))
        msce = tf.reduce_mean(squared_error, axis=dims_to_reduce)
        normalization = tf.reduce_mean(tf.square(tf.abs(y_true)), axis=dims_to_reduce)
        nrmsce = tf.math.divide_no_nan(tf.sqrt(msce), tf.sqrt(normalization))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            nrmsce = tf.multiply(nrmsce, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(nrmsce), tf.float32))
        self.total_nrmsce.assign_add(tf.reduce_sum(nrmsce))

    def result(self):
        return tf.math.divide_no_nan(self.total_nrmsce, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_nrmsce.assign(0.)
        self.count.assign(0.)


class PSNR_Complex(tf.keras.metrics.Metric):
    '''
    Metric that calculates peak signal to noise ratio given true and predicted complex valued tensors.
    '''
    def __init__(self, name='psnr_complex', max_val=None, **kwargs):
        super(PSNR_Complex, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.total_psnr = self.add_weight(name='total_psnr', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        # dimensions except batch dim must be reduced
        dims_to_reduce = tf.range(1, tf.rank(y_true))
        
        # use max values from data if not provided
        if self.max_val is None:
            max_values = tf.reduce_max(tf.abs(y_true), axis=dims_to_reduce)
        else:
            max_values = self.max_val
        
        squared_error = tf.square(tf.abs(y_true - y_pred))
        mse = tf.reduce_mean(squared_error, axis=dims_to_reduce)
        psnr = 10 * tf.math.log(tf.square(max_values)/mse) / tf.math.log(tf.constant(10, dtype=tf.float32))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            psnr = tf.multiply(psnr, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(psnr), tf.float32))
        self.total_psnr.assign_add(tf.reduce_sum(psnr))

    def result(self):
        return tf.math.divide_no_nan(self.total_psnr, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_psnr.assign(0.)
        self.count.assign(0.)


class SSIM(tf.keras.metrics.Metric):
    '''
    Metric that calculates ssim given true and predicted complex valued tensors.
    '''
    def __init__(self, name='structural_similarity_index', max_val=None, **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.total_ssim = self.add_weight(name='total_ssim', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        # dimensions except batch dim must be reduced
        dims_to_reduce = tf.range(1, tf.rank(y_true))
        
        # use max values from data if not provided
        if self.max_val is None:
            # keepdims=True is needed for tensorflow ssim function to work with 
            # different max_values per sample
            max_values = tf.reduce_max(tf.abs(y_true), axis=dims_to_reduce, keepdims=True)
            max_values = tf.expand_dims(max_values, -1)
        else:
            max_values = self.max_val

        ssim = tf.image.ssim(tf.abs(y_true[...,tf.newaxis]), tf.abs(y_pred[...,tf.newaxis]), max_val=max_values)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            ssim = tf.multiply(ssim, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(ssim), tf.float32))
        self.total_ssim.assign_add(tf.reduce_sum(ssim))

    def result(self):
        return tf.math.divide_no_nan(self.total_ssim, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_ssim.assign(0.)
        self.count.assign(0.)


class SSIM_Multiscale(tf.keras.metrics.Metric):
    '''
    Metric that calculates multiscale ssim given true and predicted complex valued tensors.
    '''
    def __init__(self, name='structural_similarity_index_multiscale', max_val=None, **kwargs):
        super(SSIM_Multiscale, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.total_ssim = self.add_weight(name='total_ssim_multiscale', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        # dimensions except batch dim must be reduced
        dims_to_reduce = tf.range(1, tf.rank(y_true))
        
        # use max values from data if not provided
        if self.max_val is None:
            # keepdims=True is needed for tensorflow ssim function to work with 
            # different max_values per sample
            max_values = tf.reduce_max(tf.abs(y_true), axis=dims_to_reduce, keepdims=True)
            max_values = tf.expand_dims(max_values, -1)
        else:
            max_values = self.max_val

        ssim = tf.image.ssim_multiscale(y_true[...,tf.newaxis], y_pred[...,tf.newaxis], max_val=max_values)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            ssim = tf.multiply(ssim, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(ssim), tf.float32))
        self.total_ssim.assign_add(tf.reduce_sum(ssim))

    def result(self):
        return tf.math.divide_no_nan(self.total_ssim, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_ssim.assign(0.)
        self.count.assign(0.)


class SSIM_Multichannel(tf.keras.metrics.Metric):
    '''
    Metric that calculates multichannel ssim given true and predicted complex valued tensors.
    '''
    def __init__(self, name='structural_similarity_index_multichannel', max_val=None, **kwargs):
        super(SSIM_Multichannel, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.total_ssim = self.add_weight(name='total_ssim_multichannel', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        # dimensions except batch dim must be reduced
        dims_to_reduce = tf.range(1, tf.rank(y_true))
        
        # use max values from data if not provided
        if self.max_val is None:
            # keepdims=True is needed for tensorflow ssim function to work with 
            # different max_values per sample
            max_values = tf.reduce_max(tf.abs(y_true), axis=dims_to_reduce, keepdims=True)
            max_values = tf.expand_dims(max_values, -1)
        else:
            max_values = self.max_val

        y_true = mrutils.complex_to_channels(y_true, axis=-1)
        y_pred = mrutils.complex_to_channels(y_pred, axis=-1)

        ssim = tf.image.ssim(y_true, y_pred, max_val=max_values)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            ssim = tf.multiply(ssim, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(ssim), tf.float32))
        self.total_ssim.assign_add(tf.reduce_sum(ssim))

    def result(self):
        return tf.math.divide_no_nan(self.total_ssim, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_ssim.assign(0.)
        self.count.assign(0.)
