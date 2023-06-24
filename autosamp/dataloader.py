from absl import app
from absl import flags


import h5py
import math
import numpy as np
import os
import glob
import autosamp.mrutils as mrutils
import tensorflow as tf
import tensorflow_addons as tfa

FLAGS = flags.FLAGS        

def read_knees(data_dir, split, data_shape=None, num_compressed_coils=None, emulate_single_coil=False, crop=True, quadratic_phase_factor=0, return_img=True, return_kspace=False):
    """
    Reads knees .tfrecords files from file as tf.data.Dataset
    
    Inputs:
        data_dir: Top directory where 'train', 'validate' and 'test' subdirectories containing tfrecords files are stored.
        split: Data split for the dataset. One of 'train', 'val' or 'test'.
        data_shape: Data shape to resize the images. No resizing is done if `data_shape` is None.
            Cropping or padding behavior is determined by the `crop` parameter.
        num_compressed_coils: Number of coils to compress the data. If None, uses all the coils (no coil compression done).
        emulate_single_coil: Whether to emulate single coil scenario by ignoring the sensitivity maps and using sense combination
            as the ground truth input. When 'True' outputs a single coil with uniform (i.e., all ones) sensitivity pattern for the
            sensitivity map.
        crop: Determines how resizing is done. If True, resizes the images to `data_shape` by either centrally cropping the image 
            or padding it evenly with zeros (uses tf.image.resize_with_crop_or_pad). If False, resizes an image to `data_shape` 
            by keeping the aspect ratio the same without distortion, hence no cropping is done (uses tf.image.resize_with_pad).
        quadratic_phase_factor: Quadratic phase factor to apply to the image doman data. If 0, no quadratic phase is applied.
            Added phase has the expression phase = `phase_factor` * pi * (y^2 + z^2) where y and z are the spatial coordinates.
    
    Returns:
        dataset: tf.data.Dataset consisting of (input, target) elements where input is a tuple in the (img_x, map_x) format.
    
    """
    
    sub_dir = 'train' if split=='train' else 'validate' if split=='val' else 'test' if split=='test' else split
    tfrecords_dir = os.path.join(data_dir, sub_dir)
    tfrecords_files = glob.glob(tfrecords_dir + '/*.tfrecords')
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_files)
    
    def _parse_and_process_knees(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                #'name': tf.FixedLenFeature([], tf.string),
                #'xslice': tf.FixedLenFeature([], tf.int64),
                #'ks_shape_x': tf.FixedLenFeature([], tf.int64),
                'ks_shape_y': tf.io.FixedLenFeature([], tf.int64),
                'ks_shape_z': tf.io.FixedLenFeature([], tf.int64),
                'ks_shape_c': tf.io.FixedLenFeature([], tf.int64),
                #'map_shape_x': tf.FixedLenFeature([], tf.int64),
                'map_shape_y': tf.io.FixedLenFeature([], tf.int64),
                'map_shape_z': tf.io.FixedLenFeature([], tf.int64),
                'map_shape_c': tf.io.FixedLenFeature([], tf.int64),
                #'map_shape_m': tf.FixedLenFeature([], tf.int64),
                'ks': tf.io.FixedLenFeature([], tf.string),
                'map': tf.io.FixedLenFeature([], tf.string)
                })
        
        # process k-space data
        ks_shape_y = tf.cast(features['ks_shape_y'], dtype=tf.int32)
        ks_shape_z = tf.cast(features['ks_shape_z'], dtype=tf.int32)
        ks_shape_c = tf.cast(features['ks_shape_c'], dtype=tf.int32)
        image_shape = [ks_shape_c, ks_shape_z, ks_shape_y]
        ks_x = tf.io.decode_raw(features['ks'], tf.float32)
        ks_x = tf.reshape(ks_x, image_shape + [2]) # 2 is due to complex values
        ks_x = mrutils.channels_to_complex(ks_x, -1)
        ks_x = tf.reshape(ks_x, image_shape)

        # process coil sensitivity maps
        map_shape_y = tf.cast(features['map_shape_y'], dtype=tf.int32)
        map_shape_z = tf.cast(features['map_shape_z'], dtype=tf.int32)
        map_shape_c = tf.cast(features['map_shape_c'], dtype=tf.int32)
        map_shape = [map_shape_c, map_shape_z, map_shape_y]
        map_x = tf.io.decode_raw(features['map'], tf.float32)
        map_x = tf.reshape(map_x, map_shape + [2]) # 2 is due to complex values
        map_x = mrutils.channels_to_complex(map_x, -1)
        map_x = tf.reshape(map_x, map_shape)

        img_x = mrutils.sense_transpose(ks_x, map_x)

        if emulate_single_coil:
            map_x = tf.ones([1, map_shape_z, map_shape_y], dtype=map_x.dtype)
        else:
            if num_compressed_coils is not None:
                mm = tf.reshape(map_x, [map_shape_c, map_shape_z * map_shape_y])
                # kk = tf.reshape(ks_x, [ks_shape_c, ks_shape_z * ks_shape_y])
                s, u, v = tf.linalg.svd(mm)
                map_x = tf.linalg.matmul(u[:, :num_compressed_coils], mm, adjoint_a=True)
                # ks_x = tf.linalg.matmul(u[:, :num_compressed_coils], kk, adjoint_a=True)
                # cc = np.matmul(mm, mm.conj().T)
                # w, v = np.linalg.eig(cc)
                # ceig = np.dot(v.conj().T, mm)
                # ceig = np.reshape(ceig, map_shape)

                map_shape_c = tf.cast(num_compressed_coils, dtype=tf.int32)
                # ks_shape_c = tf.cast(num_compressed_coils, dtype=tf.int32)
                # image_shape = [ks_shape_c, ks_shape_z, ks_shape_y]
                map_shape = [map_shape_c, map_shape_z, map_shape_y]
                map_x = tf.reshape(map_x, map_shape)
                # ks_x = tf.reshape(ks_x, image_shape)
                # coil sensitivity normalization
                map_x = tf.math.divide_no_nan(map_x, tf.reduce_sum(tf.math.conj(map_x) * map_x, axis=0, keepdims=True))

        # resize if needed
        if data_shape is not None:
            img_x = mrutils.complex_to_channels(img_x, axis=-1)
            if crop:
                img_x = tf.image.resize_with_crop_or_pad(img_x, *data_shape)
            else:
                img_x = tf.image.resize_with_pad(img_x, *data_shape)
            img_x = mrutils.channels_to_complex(img_x, axis=-1)

            map_x = mrutils.complex_to_channels(map_x, axis=-1)
            if crop:
                map_x = tf.image.resize_with_crop_or_pad(map_x, *data_shape)
            else:
                map_x = tf.image.resize_with_pad(map_x, *data_shape)
            map_x = mrutils.channels_to_complex(map_x, axis=-1)
            # coil sensitivity normalization
            map_x = tf.math.divide_no_nan(map_x, tf.reduce_sum(tf.math.conj(map_x) * map_x, axis=0, keepdims=True))

        # normalize to [0,1] range
        scale = tf.cast(tf.reduce_max(tf.abs(img_x), axis=[-2,-1], keepdims=True), tf.complex64)
        img_x = img_x / scale
        ks_x = ks_x / scale

        # add quadratic phase in image domain
        if quadratic_phase_factor != 0.0:
            img_x = mrutils.add_quadratic_phase(img_x, quadratic_phase_factor)

        if return_img:
            if return_kspace:
                inputs = (img_x, ks_x, map_x)
            else:
                inputs = (img_x, map_x)
        else:
            if return_kspace:
                inputs = (ks_x, map_x)
            else:
                inputs = map_x
        targets = img_x
        return inputs, targets # return (input, target) where input is a tuple in this case

    dataset = dataset.map(_parse_and_process_knees)
    return dataset


def read_fastmri(data_dir, split, data_shape=None):
    "Reads fastMRI .tfrecords files from file as tf.data.Dataset"

    sub_dir = 'multicoil_train' if split=='train' else 'multicoil_val' if split=='val' else 'multicoil_test' if split=='test' else split
    tfrecords_dir = os.path.join(data_dir, sub_dir)
    tfrecords_files = glob.glob(tfrecords_dir + '/*.tfrecords')
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_files)

    def _parse_and_process_fastmri(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'name': tf.io.FixedLenFeature([], tf.string),
                'slice': tf.io.FixedLenFeature([], tf.int64),
                'ks_shape_c': tf.io.FixedLenFeature([], tf.int64),
                'ks_shape_h': tf.io.FixedLenFeature([], tf.int64),
                'ks_shape_w': tf.io.FixedLenFeature([], tf.int64),
                'map_shape_c': tf.io.FixedLenFeature([], tf.int64),
                'map_shape_h': tf.io.FixedLenFeature([], tf.int64),
                'map_shape_w': tf.io.FixedLenFeature([], tf.int64),
                'ks': tf.io.FixedLenFeature([], tf.string),
                'map': tf.io.FixedLenFeature([], tf.string),
                'recon_espirit': tf.io.FixedLenFeature([], tf.string),
                'recon_rss': tf.io.FixedLenFeature([], tf.string),
                })
        
        # process coil sensitivity maps
        sensemap = tf.io.parse_tensor(features['map'], out_type=tf.complex64)

        # process espirit recon
        img = tf.io.parse_tensor(features['recon_espirit'], out_type=tf.complex64)

        # resize if needed
        if data_shape is not None:
            img = tf.image.resize_with_pad(img, *data_shape)
            sensemap = tf.image.resize_with_pad(sensemap, *data_shape)
            # coil sensitivity normalization
            sensemap = tf.math.divide_no_nan(sensemap, tf.reduce_sum(tf.math.conj(sensemap) * sensemap, axis=0, keepdims=True))

        # normalize to [0,1] range
        scale = tf.cast(tf.reduce_max(tf.abs(img), axis=[-2,-1], keepdims=True), tf.complex64)
        img = img / scale
        return (img, sensemap), img # return (input, target) where input is a tuple in this case

    dataset = dataset.map(_parse_and_process_fastmri)
    return dataset


def read_modl_brain(data_dir, split):
    """
    Reads modl brain dataset stored in .hdf5 file as tf.data.Dataset"
    
    Inputs:
        data_dir: Top directory where 'modl_data.hdf5' is stored.
        split: Data split for the dataset. One of 'train' or 'test'.
    
    Returns:
        dataset: tf.data.Dataset consisting of (input, target) elements where input is a tuple in the (img_x, map_x) format.
    
    """

    data_path = os.path.join(data_dir, 'modl_data.hdf5')

    keyname_img = 'trnOrg' if split=='train' else 'tstOrg'
    keyname_sensemap = 'trnCsm' if split=='train' else 'tstCsm'

    with h5py.File(data_path) as f:
        org = np.array(f[keyname_img], dtype=np.complex64)
        sensemap = np.array(f[keyname_sensemap], dtype=np.complex64)

    dataset = tf.data.Dataset.from_tensor_slices(((org, sensemap), org))
    return dataset


def read_pcmri(data_dir, split, num_pe=None, cropsize_readout=None, num_channels=None, num_emaps=None, augment=False): # data shape is in format (h,w)=(y,x)
    """
    Reads pcmri .tfrecords files from file as tf.data.Dataset.

    Inputs:
        data_dir: top directory for the dataset
        split: dataset split, one of 'train'/'val'/'test'
        num_pe: Number of phase encoding lines to keep. When num_pe is greater than the total
            number of phase encodes in kspace, the central num_pe phase encodes is cropped.
            If num_pe is greater than the total number of phase encodes in the image, the image
            is discarded.
        cropsize_readout: Size of central image crop in readout direction.
            Crop is applied on image domain so it reduces the FOV.
        num_channel: number of coils to use, if None uses all of the channels in datasets
        num_emaps: number of espirit maps to use, if None uses all of the emaps in dataset
        augment: boolean value indicating whether data augmentation will be done. Augmentations include
            temporal shift, spatial translation, flipping, rotations, and cropping along the read dimension

    Returns:
        dataset: tf.data.Dataset object with elements (input, target) where input is a tuple in this case
    """
    
    sub_dir = 'train' if split=='train' else 'validate' if split=='val' else 'test' if split=='test' else split
    tfrecords_dir = os.path.join(data_dir, sub_dir)
    tfrecords_files = glob.glob(tfrecords_dir + '/*.tfrecords')
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_files)

    def _parse_and_process_pcmri(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'name': tf.io.FixedLenFeature([], tf.string),
                'slice': tf.io.FixedLenFeature([], tf.int64),
                'shape_x': tf.io.FixedLenFeature([], tf.int64),
                'shape_y': tf.io.FixedLenFeature([], tf.int64),
                'shape_c': tf.io.FixedLenFeature([], tf.int64), # coils
                'shape_t': tf.io.FixedLenFeature([], tf.int64), # cardiac phases
                'shape_m': tf.io.FixedLenFeature([], tf.int64), # ESPIRiT maps
                'images': tf.io.FixedLenFeature([], tf.string), # (nx, ny, num_phases, 1, num_coils)
                'map': tf.io.FixedLenFeature([], tf.string), # (nx, ny, 1, num_emaps, num_coils)
            })
        
        # read tfrecord dimension fields
        name = features['name']
        xslice = tf.cast(features['slice'], dtype=tf.int32)
        shape_x = tf.cast(features['shape_x'], dtype=tf.int32)
        shape_y = tf.cast(features['shape_y'], dtype=tf.int32)
        shape_t = tf.cast(features['shape_t'], dtype=tf.int32)

        if num_channels is None:
            shape_c = tf.cast(features['shape_c'], dtype=tf.int32)
        else:
            shape_c = num_channels

        if num_emaps is None:
            shape_m = tf.cast(features['shape_m'], dtype=tf.int32)
        else:
            shape_m = num_emaps

        if 'shape_e' in features:
            shape_e = tf.cast(features['shape_e'], dtype=tf.int32)
        else:
            shape_e = tf.convert_to_tensor(1, dtype=tf.int32)
        
        # images
        im_x_record_bytes = tf.io.decode_raw(features['images'], tf.float32)
        image_x_shape = [shape_x, shape_y, shape_t, shape_e, shape_c]
        im_raw = tf.reshape(im_x_record_bytes, image_x_shape + [2])
        im_raw = mrutils.channels_to_complex(im_raw, axis=-1)
        im_raw = tf.reshape(im_raw, image_x_shape) # (x,y,t,e,c)

        # sensemap
        map_record_bytes = tf.io.decode_raw(features['map'], tf.float32)
        map_shape = [shape_x, shape_y, shape_m, shape_c]
        map_x = tf.reshape(map_record_bytes, map_shape + [2])
        map_x = mrutils.channels_to_complex(map_x, axis=-1)
        map_x = tf.reshape(map_x, map_shape) # (x,y,m,c)

        # data augmentation
        if augment:
            # temporal (and circular) shift
            shifts = tf.range(-3, 4, dtype=tf.int64)
            shifts = tf.random_shuffle(shifts)
            im_raw = tf.roll(im_raw, shifts[0], axis=2)

            # squash non-spatial dims before augmentation
            im_squash = tf.reshape(im_raw, [shape_x, shape_y, shape_t*shape_e*shape_c])
            map_squash = tf.reshape(map_x, [shape_x, shape_y, shape_m*shape_c])
            all_squash = tf.concat([im_squash, map_squash], axis=2)

            # flip (X,Y)
            all_squash = tf.image.random_flip_left_right(all_squash)
            all_squash = tf.image.random_flip_up_down(all_squash)

            # spatial (circular) translation in Y
            shift_max = tf.cast(0.15*tf.to_float(shape_y), dtype=tf.int64) # +/- 15%
            shifts = tf.range(-shift_max, shift_max, dtype=tf.int64)
            shifts = tf.random.shuffle(shifts)
            all_squash = tf.roll(all_squash, shifts[0], axis=1)

            # rotate X/Y
            num_angles = 128 # number of rot angles
            rot_angles = tf.range(0, 2*math.pi, delta=2*math.pi/num_angles, dtype=tf.float32)
            rot_angles = tf.random.shuffle(rot_angles)
            # note: need to pass real data to rotate fn
            all_rot = mrutils.complex_to_channels(all_squash, axis=0) # put in batch dim
            all_rot = tfa.image.rotate(all_rot, rot_angles[0], interpolation='bilinear')
            all_squash = mrutils.channels_to_complex(all_rot, axis=0)

            # centrally-focused crops (X)
            ctr = tf.cast(tf.floor(shape_x/2) + 1, dtype=tf.float32)
            patch_ctr = tf.cast(tf.floor(cropsize_readout/2) + 1, dtype=tf.float32)
            sig = 0.5 * (ctr-patch_ctr-1)
            crop_ctr = tf.truncated_normal([], mean=ctr, stddev=sig)
            crop_start = tf.cast(crop_ctr - tf.floor(cropsize_readout/2) + 1, dtype=tf.int64)
            all_squash = tf.slice(all_squash, [crop_start,0,0], [cropsize_readout,-1,-1])
            shape_x = cropsize_readout

            # Un-squash spatial dims of augmented data
            im_raw = tf.slice(all_squash, [0 ,0, 0], [-1, -1, shape_t*shape_c*shape_e]) # (x,y,t*e*c)
            map_x = tf.slice(all_squash, [0, 0, shape_t*shape_c*shape_e], [-1, -1, shape_m*shape_c]) # (x,y,m*c)

        # reshape to full dims
        im_raw = tf.reshape(im_raw, [shape_x, shape_y, shape_t, shape_e, 1, shape_c]) # (x,y,t,e,m,c)
        map_x = tf.reshape(map_x, [shape_x, shape_y, 1, 1, shape_m, shape_c]) # (x,y,t,e,m,c)
        # put spatial dims to end (innermost dims)
        im_raw = tf.transpose(im_raw, [3,4,5,2,1,0]) # (e,m,c,t,y,x)
        map_x = tf.transpose(map_x, [3,4,5,2,1,0]) # (e,m,c,t,y,x)
        ks_truth = mrutils.fft2c(im_raw)
        map_ks = mrutils.fft2c(map_x)
        if num_pe is not None:
            if num_pe <= shape_y:
                # remove the extra phase encodes
                ks_truth = mrutils.center_crop(ks_truth, [num_pe, shape_x])
                map_ks = mrutils.center_crop(map_ks, [num_pe, shape_x])
                shape_y = num_pe
                im_raw = mrutils.ifft2c(ks_truth)
                map_x = mrutils.ifft2c(map_ks)
        
        if cropsize_readout is not None:
            # crop image in read-encode direction
            im_raw = mrutils.center_crop(im_raw, [shape_y, cropsize_readout])
            map_x = mrutils.center_crop(map_x, [shape_y, cropsize_readout])
            shape_x = cropsize_readout
            ks_truth = mrutils.fft2c(im_raw)

        # coil sensitivity normalization
        map_scale = tf.reduce_sum(tf.abs(map_x*tf.math.conj(map_x)), axis=-1, keepdims=True)
        map_scale = tf.sqrt(tf.reduce_sum(tf.math.conj(map_x) * map_x, axis=-3, keepdims=True))
        map_x = tf.math.divide_no_nan(map_x, map_scale) # (e,m,c,t,y,x)
        map_ks = mrutils.fft2c(map_x)

        im_truth = mrutils.sense_transpose(ks_truth, map_x, channel_axis=-4) # (e,m,t,y,x)

        # compute scaling factor from time-averaged recon (95% max magnitude)
        im_avg = tf.reduce_mean(im_truth, axis=-3, keepdims=True)
        im_avg = tf.abs(im_avg)
        k = tf.cast(0.05 * tf.cast(tf.size(im_avg), dtype=tf.float32), dtype=tf.int32)
        top_k_vals, _ = tf.math.top_k(tf.reshape(im_avg, [-1]), k=k)
        scale_x = tf.reduce_min(top_k_vals)

        # Apply scaling
        scale_x = tf.cast(scale_x, dtype=tf.complex64)
        ks_truth = tf.math.divide_no_nan(ks_truth, scale_x)
        im_truth = tf.math.divide_no_nan(im_truth, scale_x)

        # return (input, target) where input is a tuple in this case
        return (im_truth[0], map_x[0]), im_truth[0] # im_truth: (m,t,y,x), map_x: (m,c,t=1,y,x)

    dataset = dataset.map(_parse_and_process_pcmri)
    # filter out the scans with smaller phase encodes
    if num_pe is not None:
        dataset = dataset.filter(lambda x,y: tf.shape(y)[-2] == num_pe)
    
    return dataset