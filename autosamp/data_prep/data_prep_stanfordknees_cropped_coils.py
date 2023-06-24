from dataclasses import asdict
from fileinput import filename
import os
import glob
from tqdm import tqdm
import subprocess
import argparse

import mridata
import tensorflow as tf
import ismrmrd
import numpy as np
import sigpy.mri

import autosamp.data_prep.utils.logging
from autosamp.data_prep.utils import tfmri
from autosamp.data_prep.utils import fftc
from autosamp.data_prep.utils import mri

logger = autosamp.data_prep.utils.logging.logger

import sys
import os
sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
from bart import bart

# there is duplicate data in mridata knees dataset
duplicate_data_id = 'b7d435a1-2421-48d2-946c-d1b3372f7c60'

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def setup_data_tfrecords(dir_input,
                         dir_output,
                         dir_test_npy=None,
                         test_acceleration=12,
                         test_calib=20,
                         data_divide=(.75, .05, .2),
                         num_compressed_coils=None):
    """Setups training data as tfrecords."""
    logger.info('Converting npy data to TFRecords in {}...'.format(dir_output))

    file_list = glob.glob(dir_input + '/*.npy')
    file_list = [os.path.basename(f) for f in file_list]
    file_list = sorted(file_list)
    num_files = len(file_list)

    i_train_1 = np.round(data_divide[0] * num_files).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(
        data_divide[1] * num_files).astype(int) + i_validate_0

    if not os.path.exists(os.path.join(dir_output, 'train')):
        os.makedirs(os.path.join(dir_output, 'train'))
    if not os.path.exists(os.path.join(dir_output, 'validate')):
        os.makedirs(os.path.join(dir_output, 'validate'))
    if not os.path.exists(os.path.join(dir_output, 'test')):
        os.makedirs(os.path.join(dir_output, 'test'))

    # if dir_test_npy:
        # if not os.path.exists(dir_test_npy):
            # os.makedirs(dir_test_npy)

    max_shape_y, max_shape_z = 0, 0

    for i_file, file_name in enumerate(file_list):
        if duplicate_data_id not in file_name:
            # testing = False
            if i_file < i_train_1:
                dir_output_i = os.path.join(dir_output, 'train')
            elif i_file < i_validate_1:
                dir_output_i = os.path.join(dir_output, 'validate')
            else:
                # testing = True
                dir_output_i = os.path.join(dir_output, 'test')

            logger.info('Processing [%d] %s...' % (i_file, file_name))

            file_kspace = os.path.join(dir_input, file_name)
            kspace = np.squeeze(np.load(file_kspace))
            file_name_noext = os.path.splitext(file_name)[0]

            shape_x = kspace.shape[-1]
            shape_y = kspace.shape[-2]
            shape_z = kspace.shape[-3]
            shape_c = kspace.shape[-4]
            if shape_y > max_shape_y:
                max_shape_y = shape_y
            if shape_z > max_shape_z:
                max_shape_z = shape_z
            logger.debug('  Slice shape: (%d, %d)' % (shape_z, shape_y))
            logger.debug('  Num channels: %d' % shape_c)

            # if testing and dir_test_npy:
            #     logger.info(
            #         '  Creating npy test data (R={})...'.format(test_acceleration))
            #     logger.debug('    Generating sampling mask...')
            #     random_seed = 1e6 * np.random.random_sample()
            #     mask = sigpy.mri.poisson([shape_z, shape_y],
            #                              test_acceleration,
            #                              calib=[test_calib] * 2,
            #                              seed=random_seed)
            #     mask = np.reshape(mask, [1, shape_z, shape_y, 1])

            #     logger.debug('    Applying sampling mask...')
            #     kspace_test = kspace.copy() * mask
            #     file_kspace_out = os.path.join(
            #         dir_test_npy,
            #         file_name_noext + '_R{}.npy'.format(test_acceleration))
            #     logger.debug('    Writing file {}...'.format(file_kspace_out))
            #     np.save(file_kspace_out, kspace_test.astype(np.complex64))

            #     file_kspace_out = os.path.join(dir_test_npy,
            #                                    file_name_noext + '_truth.npy')
            #     np.save(file_kspace_out, kspace.astype(np.complex64))

            # logger.info('  Estimating sensitivity maps...')
            # sensemap = mri.estimate_sense_maps(kspace, calib=test_calib)
            # sensemap = np.expand_dims(sensemap, axis=0)

            logger.info('  Creating tfrecords (%d)...' % shape_x)
            kspace = fftc.ifftc(kspace, axis=-1)
            kspace = kspace.astype(np.complex64)
            for i_x in range(shape_x):
                file_out = os.path.join(
                    dir_output_i, '%s_x%03d.tfrecords' % (file_name_noext, i_x))
                
                if not os.path.exists(file_out):
                    kspace_x = kspace[:, :, :, i_x]
                    # sensemap_x = sensemap[:, :, :, :, i_x]
                    try:
                        sensemap_x = mri.estimate_sense_maps(kspace_x[:,None,:,:], calib=test_calib, autothresh=True)
                        sensemap_x = np.expand_dims(np.squeeze(sensemap_x), axis=0)
                    except Exception as e:
                        logger.error('  Error in sensitivity calculation for %s' % file_out)
                    else:
                        # compress the coils
                        if num_compressed_coils is not None:
                            cmatrix = bart(1, 'bart cc -p %d -S -r %d -M' % (num_compressed_coils, test_calib), np.transpose(kspace_x)[:,:,None,:])
                            cmatrix = np.squeeze(cmatrix)
                            sensemap_x = np.tensordot(cmatrix[:,:num_compressed_coils].conj().T, sensemap_x, axes=[[1],[1]])
                            sensemap_x = np.transpose(sensemap_x, axes=[1,0,2,3,4])
                            # coil sensitivity normalization
                            scaling = np.sum(np.abs(sensemap_x)**2, axis=[1], keepdims=True)
                            sensemap_x = np.divide(sensemap_x, scaling, where=scaling!=0.)

                        kspace_x = kspace_x.astype(np.complex64)
                        sensemap_x = sensemap_x.astype(np.complex64)
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'name': _bytes_feature(str.encode(file_name_noext)),
                                    'xslice': _int64_feature(i_x),
                                    'ks_shape_x': _int64_feature(kspace.shape[3]),
                                    'ks_shape_y': _int64_feature(kspace.shape[2]),
                                    'ks_shape_z': _int64_feature(kspace.shape[1]),
                                    'ks_shape_c': _int64_feature(kspace.shape[0]),
                                    'map_shape_x': _int64_feature(kspace.shape[3]),
                                    'map_shape_y': _int64_feature(sensemap_x.shape[3]),
                                    'map_shape_z': _int64_feature(sensemap_x.shape[2]),
                                    'map_shape_c': _int64_feature(sensemap_x.shape[1]),
                                    'map_shape_m': _int64_feature(sensemap_x.shape[0]),
                                    # 'ks': _bytes_feature(tf.io.serialize_tensor(kspace_x)),
                                    # 'map': _bytes_feature(tf.io.serialize_tensor(sensemap_x))
                                    'ks': _bytes_feature(kspace_x.tostring()),
                                    'map': _bytes_feature(sensemap_x.tostring())
                                }))

                        tf_writer = tf.io.TFRecordWriter(file_out)
                        tf_writer.write(example.SerializeToString())
                        tf_writer.close()

    return max_shape_z, max_shape_y




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation')
    # parser.add_argument(
        # 'mridata_txt',
        # action='store',
        # help='Text file with mridata.org UUID datasets')
    parser.add_argument(
        '--output',
        default='data',
        help='Output root directory (default: data)')
    parser.add_argument(
        '--dir_npy',
        help='Directory where input npy files are saved.')
    parser.add_argument(
        '--num_compressed_coils',
        type=int,
        help='Number of virtual coils to use. No compression is applied if None',
        default=None)
    parser.add_argument('--random_seed', default=1000, help='Random seed')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='verbose printing (default: False)')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(autosamp.data_prep.utils.logging.logging.DEBUG)

    if args.random_seed >= 0:
        np.random.seed(args.random_seed)

    # dir_mridata_org = os.path.join(args.output, 'raw/ismrmrd')
    # download_mridata_org_dataset(args.mridata_txt, dir_mridata_org)

    # dir_npy = os.path.join(args.output, 'raw/npy')
    # ismrmrd_to_npy(dir_mridata_org, dir_npy)

    if args.num_compressed_coils is None:
        dir_tfrecord = os.path.join(args.output, 'tfrecord_croppedmap')
    else:
        dir_tfrecord = os.path.join(args.output, 'tfrecord_croppedmap_cc%d' % args.num_compressed_coils)
    # dir_test_npy = os.path.join(args.output, 'test_npy')
    shape_z, shape_y = setup_data_tfrecords(
        args.dir_npy, dir_tfrecord)

    # dir_masks = os.path.join(args.output, 'masks')
    # create_masks(dir_masks, shape_z=shape_z, shape_y=shape_y, num_repeat=48)
