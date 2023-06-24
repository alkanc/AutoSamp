import os
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--dir_npy',
    help='Directory where input npy files are saved.')
    parser.add_argument(
        '--output',
        default='data',
        help='Output root directory (default: data)')

    args = parser.parse_args()
    dir_input = args.dir_npy
    dir_output = args.output

    data_divide=(.75, .05, .2)

    file_list = glob.glob(dir_input + '/*.npy')
    file_list = [os.path.basename(f) for f in file_list]
    file_list = sorted(file_list)
    num_files = len(file_list)


    i_train_1 = np.round(data_divide[0] * num_files).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(
        data_divide[1] * num_files).astype(int) + i_validate_0

    file_list = [elem.split('.npy')[0] for elem in file_list]


    dir_train = os.path.join(dir_output, 'train')
    dir_validate = os.path.join(dir_output, 'validate')
    dir_test = os.path.join(dir_output, 'test')

    tfrecords_files = glob.glob(dir_train + '/*.tfrecords') \
                      + glob.glob(dir_validate + '/*.tfrecords') \
                      + glob.glob(dir_test + '/*.tfrecords') \
    
    print(len(tfrecords_files))
    
    for ff in tqdm(tfrecords_files):
        ff_name = os.path.basename(ff)
        ff_name = ff_name.split('_x')[0]
        i_file = file_list.index(ff_name)

        if i_file < i_train_1:
            dir_output_i = os.path.join(dir_output, 'train')
        elif i_file < i_validate_1:
            dir_output_i = os.path.join(dir_output, 'validate')
        else:
            dir_output_i = os.path.join(dir_output, 'test')        

        try:
            shutil.move(ff, dir_output_i)
        except Exception:
            pass
