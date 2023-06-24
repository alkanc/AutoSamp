"""
Adds sensitivity maps to fastMRI dataset
"""

import os, sys
import glob
import h5py
import numpy as np
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import train_test_split

import tensorflow as tf
import autosamp.mrutils as mrutils

# add BART modules
toolbox_path = os.environ["TOOLBOX_PATH"]
sys.path.append(os.path.join(toolbox_path, 'python'))
import bart, cfl  # pylint: disable=import-error

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

def main():
    parser = argparse.ArgumentParser(description='Data preparation for fastMRI knees')
    parser.add_argument(
        'input_data_path',
        help='input data path that contains the train/val/test files for fastMRI knees dataset'
    )
    parser.add_argument(
        'output_data_path',
        help='output data path to store tfRecords files'
    )
    parser.add_argument(
        '--center_fraction',
        default=0.04,
        help='number of k-space lines used to do ESPIRiT calib'
    )
    parser.add_argument(
        '--random_seed',
        default=82,
        help='random seed to use for splitting val dataset into val and test'
    )
    parser.add_argument(
        '--test_perc',
        default=0.5,
        help='percentage of validation dataset to be used for test dataset'
    )

    args = parser.parse_args()

	# ARGS
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    center_fraction = float(args.center_fraction)
    random_seed = int(args.random_seed)
    test_perc = float(args.test_perc)
    num_emaps = 1
	# dbwrite = False

    train_files = glob.glob(os.path.join(input_data_path, 'multicoil_train', '*.h5'))
    val_files = glob.glob(os.path.join(input_data_path, 'multicoil_val', '*.h5'))
    val_files, test_files = train_test_split(val_files, test_size=test_perc, random_state=random_seed)
	
    output_folders = ['multicoil_train', 'multicoil_val', 'multicoil_test']
    split_files = [train_files, val_files, test_files]


    for folder, input_files in zip(output_folders, split_files):
        print('Processing %s'%folder)
        folder_path = os.path.join(output_data_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
	
        for h5file in input_files:
            # Load HDF5 file
            hf = h5py.File(h5file, 'r')
            # existing keys: ['ismrmrd_header', 'kspace', 'reconstruction_rss']

            # load k-space and image data from HDF5 file
            kspace_orig = hf['kspace'][()] 
            im_rss = hf['reconstruction_rss'][()] # (33, 320, 320)

            # get data dimensions
            num_slices, num_coils, num_kx, num_ky = kspace_orig.shape
            xres, yres = im_rss.shape[1:3] # matrix size
            num_low_freqs = int(round(center_fraction * yres))

            # allocate memory for new arrays
            im_shape = (xres, yres)

            for sl in range(num_slices):
                kspace_slice = kspace_orig[np.newaxis, sl]

                # Data dimensions for BART:
                #  kspace - (kx, ky, 1, coils) 
                #  maps - (kx, ky, 1, coils, emaps)
                # Data dimensions for TF (we use channels first):
                #  kspace - (1, coils, kx, ky, real/imag)
                #  maps   - (1, coils, kx, ky, emaps, real/imag)

                # Pre-process k-space data (TensorFlow)
                image_tensor = mrutils.ifft2c(kspace_slice)
                image_tensor = mrutils.center_crop(image_tensor, im_shape)
                kspace_tensor = mrutils.fft2c(image_tensor)  # (1, 15, 320, 320)
                
                kspace_slice_bart = np.transpose(kspace_tensor.numpy(), axes=[2, 3, 0, 1])  # (320, 320, 1, 15)

                # Compute sensitivity maps (BART)
                maps_slice_bart = bart.bart(1, f'ecalib -d 0 -m {num_emaps} -c 0.2 -r {num_low_freqs}', kspace_slice_bart)
                maps_slice_bart = np.reshape(maps_slice_bart, (xres, yres, 1, num_coils, num_emaps))[...,0]  # (320, 320, 1, 15)

                maps_tensor = tf.transpose(maps_slice_bart, perm=[2, 3, 0, 1])  # (1, 15, 320, 320)

                # Do coil combination using sensitivity maps (TensorFlow)
                im_tensor = mrutils.sense_transpose(kspace_tensor, maps_tensor)  # (1, 320, 320)

                # # Convert image tensor to numpy array
                # im_slice = im_tensor.numpy()

                # Re-shape before saving
                kspace_sl = kspace_tensor[0]
                maps_sl = maps_tensor[0]
                im_truth_sl = im_tensor[0]
                im_rss_sl = im_rss[sl]

                # Save each slice as tfrecord file
                file_name_noext = '%s_sl%.3d' % (os.path.splitext(os.path.basename(h5file))[0], sl)

                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'name': _bytes_feature(str.encode(file_name_noext)),
                            'slice': _int64_feature(sl),
                            'ks_shape_c': _int64_feature(kspace_sl.shape[0]),
                            'ks_shape_h': _int64_feature(kspace_sl.shape[1]),
                            'ks_shape_w': _int64_feature(kspace_sl.shape[2]),
                            'map_shape_c': _int64_feature(maps_sl.shape[0]),
                            'map_shape_h': _int64_feature(maps_sl.shape[1]),
                            'map_shape_w': _int64_feature(maps_sl.shape[2]),
                            'ks': _bytes_feature(tf.io.serialize_tensor(kspace_sl)),
                            'map': _bytes_feature(tf.io.serialize_tensor(maps_sl)),
                            'recon_espirit': _bytes_feature(tf.io.serialize_tensor(im_truth_sl)),
                            'recon_rss': _bytes_feature(tf.io.serialize_tensor(im_rss_sl)),
                        }
                    )
                )

                # write the slice example to a TFRecord file
                file_output_path = os.path.join(output_data_path, folder, file_name_noext + '.tfrecords')
                with tf.io.TFRecordWriter(file_output_path) as writer:
                    writer.write(tf_example.SerializeToString())
                print('\t%s saved.' % file_output_path)


if __name__ == '__main__':
    main()