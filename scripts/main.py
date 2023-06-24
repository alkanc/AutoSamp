from absl import app
from absl import flags

import datetime
import glob
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import autosamp.dataloader as dataloader
import autosamp.model as models
import autosamp.mrmetrics as mrmetrics

FLAGS = flags.FLAGS

# General options
flags.DEFINE_integer('seed', 92, 'random seed for to be used for random operations')

# File options
flags.DEFINE_string('exp_dir', 'experiments', 'top directory for checkpoints, events files and results')
flags.DEFINE_string('out_dir', None, 'if specified, overwrites exp_dir. actual directory for checkpoints, events files and results')
flags.DEFINE_string('exp_name', None, 'experiment name')
flags.mark_flag_as_required('exp_name')
flags.DEFINE_string('data_dir', 'data/tfrecord', 'directory of dataset to be used')
flags.DEFINE_string('data_source', 'knees', 'knees/knees_full/fastmri/fastmri_small/modl_brain')
flags.DEFINE_list('data_shape', None, 'comma seperated list of integers for data shape. No resizing is done if None.')
flags.DEFINE_integer('num_compressed_coils', None, 'Number of compressed coils to use. No coil compression is applied if None.')
flags.DEFINE_boolean('emulate_single_coil', False, 'Emulate single coil by using coil combined image as ground truth and eliminating sensitivity maps.')
flags.DEFINE_float('quadratic_phase_factor', 0, 'quadratic phase factor to be element-wise multiplied in the image domain. Gets multiplied by pi * (y^2 + z^2)')
flags.DEFINE_boolean('simulate_from_kspace', False, 'simulate from kspace. If True, the dataloader outputs as kspace as input, and the simulated kspace is obtained directly from the kspace data without coil combination. \
                                                     If False, the dataloader outputs coil-combined image as input, and the simulated kspace is obtained by applying sensitivity maps to the coil-combined image.\
                                                     This flag is relevant only when `data_source` is `knees_full`.')

# Training options
flags.DEFINE_string('mode', 'train', 'Available modes: train/eval')
flags.DEFINE_integer('num_epochs', 100, 'number of training epochs')
flags.DEFINE_integer('batch_size', 32, 'number of datapoints per batch')
flags.DEFINE_string('optimizer', 'adam', 'sgd, adam, adamax')
flags.DEFINE_float('lr', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('lr_enc', None, 'learning rate for optimizer for the encoder. If None, uses the optimizer specified with the learning rate `--lr` is used for both encoder and decoder')
flags.DEFINE_string('loss', 'mse', 'loss function to use: mse (l2) for Gaussian observation model, \
                                    mae (l1) for Laplacian observation model')
flags.DEFINE_integer('max_checkpoints', None, 'maximum number of checkpoints to keep during training')
flags.DEFINE_boolean('single_batch_training', False, 'Train on a single batch for debugging purposes')
flags.DEFINE_integer('eval_every', 1, 'number of epochs to wait for each evaluation step')
flags.DEFINE_integer('save_ckpt_every', 1, 'number of epochs to wait for each checkpoint save')

# Model options
flags.DEFINE_list('undersamp_ratio', ['0.3'], 'undersampling ratio' ) # overwritten if '--fixed_mask_path' is specified (see below)
flags.register_validator('undersamp_ratio',
                         lambda values: all([(float(value) > 0) and (float(value) < 1) for value in values]),
                         message='--undersamp_ratio values must be in (0,1) range')
flags.DEFINE_float('noise_std', 0.1, 'std. of noise')
flags.DEFINE_string('enc_arch', 'tfnufft', 'matrix/matrix1d/matrix_jmodl_2d/fft/tfnufft') 
    # matrix: matrix of complex exponentials for encoding
    # matrix1d: matrix nudft for phase encodes, fft for freq encodes on 2d cartesian trajectory
    # matrix_jmodl_2d: jmodl type of encoding where encoding is represented in horizontal and vertical directions in a seperable fashion (results in grid like patterns)
    # fft: fft (instead of tfnufft) that works with cartesian coordinates, and requires specifying a sampling mask
    # tfnufft: tfnufft library with no density compensation
flags.DEFINE_boolean('use_dcomp', False, 'calculate and use density compensation before doing adjoint nufft')
flags.DEFINE_float('tfnufft_oversamp', 1.25, 'oversampling factor for tfnufft operations')
flags.DEFINE_float('tfnufft_width', 4., 'interpolation kernel full-width in terms of oversampled grid for tfnufft operations')
flags.DEFINE_string('coords_initializer', 'uniform', 'initializer for k-space coordinates: supports uniform/circular_uniform/small_circular_uniform/smaller_circular_uniform/normal/mask/mask_coordinates/') # mask: encoding from specified mask
                                                                                                                                                                                                            #      (requires specifying fixed_mask_path)
flags.DEFINE_integer('coords_calib_size', None, 'size of calibration region for k-space coordinates. Relevant only for `enc_arch` == tfnufft and \
                                                uniform/circular_uniform/normal/small_circular_uniform/smaller_circular_uniform initilization. \
                                                Does not use calibration region if None.')
flags.DEFINE_string('fixed_mask_path', None, 'path for the sampling mask for coordinate initializion, overwrites undersamp_ratio and coords_initializer') # overwrites undersamp_ratio and coords_initializer specified
flags.DEFINE_string('fixed_mask_coordinates_path', None, 'path for the sampling mask coordinates for coordinate initializion, overwrites coords_initializer') # overwrites coords_initializer specified
flags.DEFINE_boolean('optimize_sampling', True, 'optimize k-space sampling coordinates')
flags.DEFINE_enum('fft_mask_type', None, ['random1d', 'random2d', 'equispaced1d', 'equispacedfraction1d', 'poisson', 'fixed_mask'],
                  'sampling mask type to use for enc_arch==fft')

flags.DEFINE_integer('fft_mask_train_seed', None, 'seed for sampling mask used for training, if None then a random mask is sampled for each train iteration')
flags.DEFINE_integer('fft_mask_val_seed', None, 'seed for sampling mask used for validation, if None then a random mask is sampled for each val iteration')
flags.DEFINE_list('fft_mask_center_fractions', None, 'Fraction of low-frequency samples to be retained')
flags.DEFINE_list('fft_mask_calib_sizes', None, 'Actual size of calibration region. Either calib_sizes or center_fractions must be specified.')
flags.DEFINE_string('dec_arch', 'unrolled', 'unrolled/unet')
flags.DEFINE_integer('unrolled_steps', 4, 'number of grad steps for unrolled algorithms')
flags.DEFINE_integer('unrolled_num_resblocks', 3, 'number of ResBlocks per iteration')
flags.DEFINE_integer('unrolled_num_filters', 128, 'number of features for the convolutional layers in each ResBlock')
flags.DEFINE_integer('unrolled_kernel_size', 3, 'kernel size for the convolutional layers in each ResBlock')
flags.DEFINE_string('unrolled_act_fun', 'relu', 'activation function to use in unrolled network: relu/lrelu')
flags.DEFINE_string('unrolled_normalization_layer', 'instance', 'normalization layer to use: instance/layer/batch/None')
flags.DEFINE_boolean('unrolled_share_proxresblocks', False, 'share the ProxResBlocks across iterations')
flags.DEFINE_integer('unet_num_features', 32, 'number of features in the first level of unet')
flags.DEFINE_integer('unet_num_levels', 4, 'number of levels in unet')
flags.DEFINE_integer('unet_kernel_size', 3, 'convolution kernel size in unet')
flags.DEFINE_string('unet_normalization_layer', 'instance', 'normalization layer to use: instance/layer/batch')
flags.DEFINE_string('unet_act_fun', 'relu', 'activation function to use in unet: relu/lrelu')

def main(argv):
    
    # set random seed
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # convert undersamp_ratio to float list and make sure it's valid
    FLAGS.undersamp_ratio = [float(elem) for elem in FLAGS.undersamp_ratio]
    if len(FLAGS.undersamp_ratio) != 1 and FLAGS.enc_arch != 'fft':
        raise ValueError('multiple undersamp_ratios is only supported when enc_arch == fft')

    # convert data shape to int list
    if FLAGS.data_shape is not None:
        FLAGS.data_shape = [int(elem) for elem in FLAGS.data_shape]
    
    # convert fft_mask_center_fractions to int list
    if FLAGS.fft_mask_center_fractions is not None:
        FLAGS.fft_mask_center_fractions = [float(elem) for elem in FLAGS.fft_mask_center_fractions]
    # convert fft_mask_calib_sizes to int list
    if FLAGS.fft_mask_calib_sizes is not None:
        FLAGS.fft_mask_calib_sizes = [int(elem) for elem in FLAGS.fft_mask_calib_sizes]

    # make necessary directories
    if FLAGS.out_dir is None:
        subpath = 'noise_' + str(FLAGS.noise_std)
        FLAGS.out_dir = os.path.join(FLAGS.exp_dir, FLAGS.data_source, subpath, FLAGS.exp_name)
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir, exist_ok=True)
    
    # make sure fixed_mask_path is provided when coords_initializer is 'mask'
    if FLAGS.coords_initializer == 'mask':
        if FLAGS.fixed_mask_path is None:
            raise ValueError('fixed_mask_path must be provided when coords_initializer is \'mask\'')
    
    if FLAGS.coords_initializer == 'mask_coordinates':
        if FLAGS.fixed_mask_coordinates_path is None:
            raise ValueError('fixed_mask_coordinates_path must be provided when coords_initializer is \'mask_coordinates\'')   
    
    # make sure 'coords_initializer' and 'enc_arch' are consistent
    if FLAGS.coords_initializer == 'circular_uniform' or FLAGS.coords_initializer == 'small_circular_uniform' or FLAGS.coords_initializer == 'smaller_circular_uniform' or FLAGS.coords_initializer == 'half_circle_border':
        if FLAGS.enc_arch == 'matrix1d':
            raise ValueError('circular_uniform or small_circular_uniform or smaller_circular_uniform or half_circle_border initialization is not supported when enc_arch is matrix1d.')

    
    if FLAGS.enc_arch == 'matrix_jmodl_2d':
        if FLAGS.coords_initializer != 'uniform':
            raise ValueError('%s initialization is not supported with ench_arch=matrix_jmodl_2d' % FLAGS.coords_initializer)

    # make sure `coords_calib_size` is only specified when (`enc_arch` == tfnufft or `enc_arch` == matrix_jmodl_2d) and \
    # uniform/circular_uniform/normal/small_circular_uniform/smaller_circular_uniform initilization
    if FLAGS.coords_calib_size is not None:
        if not (FLAGS.enc_arch == 'tfnufft' or FLAGS.enc_arch == 'matrix_jmodl_2d'):
            raise ValueError('coords_calib_size is relevant only for enc_arch==tfnufft or enc_arch==matrix_jmodl_2d')
        else:
            if not (FLAGS.coords_initializer == 'uniform' or FLAGS.coords_initializer == 'circular_uniform' 
                    or FLAGS.coords_initializer == 'normal' or FLAGS.coords_initializer == 'small_circular_uniform' 
                    or FLAGS.coords_initializer == 'smaller_circular_uniform'):
                
                raise ValueError('coords_calib_size should not be specified when coords_initializer==%s' % FLAGS.coords_initializer)
    
    # make sure relevant flags are specified when fft encoding is used
    if FLAGS.enc_arch == 'fft':
        if FLAGS.fft_mask_type is None:
            raise ValueError("fft_mask_type must be specified for enc_arch==fft.")
        elif FLAGS.fft_mask_type == 'fixed_mask':
            if FLAGS.fixed_mask_path is None:
                raise ValueError('--fixed_mask_path must be specified when --fft_mask_type==fixed_mask')
        else:
            if not FLAGS.fft_mask_calib_sizes and not FLAGS.fft_mask_center_fractions:
                raise ValueError("Either fft_mask_calib_sizes or fft_mask_center_fractions must be specified.")
            if FLAGS.fft_mask_calib_sizes and FLAGS.fft_mask_center_fractions:
                raise ValueError("Only one of calib_sizes or center_fractions can be specified")
        if FLAGS.optimize_sampling:
            raise ValueError('Sample optimization is not supported when enc_arch==fft')
    
    # coil compression and single coil emulation only supported for knees for now
    if FLAGS.num_compressed_coils is not None and not (FLAGS.data_source == 'knees' or FLAGS.data_source == 'knees_full'):
        raise ValueError('Coil compression is only supported when data_source==''knees'' or  data_source==''knees_full''')
    if FLAGS.emulate_single_coil and not (FLAGS.data_source == 'knees' or FLAGS.data_source == 'knees_full'):
        raise ValueError('Single coil emulation is only supported when data_source==''knees'' or  data_source==''knees_full''')
    
    # simulate_from_kspace only supported for knees_full for now
    if FLAGS.simulate_from_kspace and not (FLAGS.data_source == 'knees' or FLAGS.data_source == 'knees_full'):
        raise ValueError('simulate_from_kspace is only supported when data_source==`knees` or  data_source==`knees_full`')

    # load the initialization sampling pattern if fixed_mask_path is specified 
    # and update undersamp_ratio and coords_initializer accordingly
    if FLAGS.fixed_mask_path is not None:
        mask = np.load(FLAGS.fixed_mask_path)
        FLAGS.undersamp_ratio = [np.sum(mask!=0) / mask.size]
        if FLAGS.enc_arch == 'fft':
            FLAGS.fft_mask_type = 'fixed_mask'
        else:
            FLAGS.coords_initializer = 'mask'
        mask_savepath = os.path.join(FLAGS.out_dir, 'sample_mask.npy')
        np.save(mask_savepath, mask)

    # load the initialization sampling pattern if fixed_mask_coordinates_path is specified 
    # and update coords_initializer accordingly
    if FLAGS.fixed_mask_coordinates_path is not None:
        mask = np.load(FLAGS.fixed_mask_coordinates_path)
        FLAGS.coords_initializer = 'mask_coordinates'
        mask_savepath = os.path.join(FLAGS.out_dir, 'sample_mask_coordinates.npy')
        np.save(mask_savepath, mask)
    
    # save configs as json and flagfile
    with open(os.path.join(FLAGS.out_dir, 'configs.json'), 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp, indent=4, separators=(',', ': '))
    with open(os.path.join(FLAGS.out_dir, 'flagfile.txt'), 'wt') as ft:
        ft.write(FLAGS.flags_into_string())

    if FLAGS.mode == 'train':
        # summary directories
        # current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_dir_train = os.path.join(FLAGS.out_dir, 'logs', 'train')
        summary_dir_val = os.path.join(FLAGS.out_dir, 'logs', 'val')
        if not os.path.exists(summary_dir_train):
            os.makedirs(summary_dir_train, exist_ok=True)
        if not os.path.exists(summary_dir_val):
            os.makedirs(summary_dir_val, exist_ok=True)

        # checkpoint directories
        ckpt_dir = os.path.join(FLAGS.out_dir, 'checkpoints')
        best_ckpt_dir = os.path.join(ckpt_dir, 'best_checkpoints')
        if not os.path.exists(best_ckpt_dir):
            os.makedirs(best_ckpt_dir, exist_ok=True)

        # output image directory (for coordinate plots)
        image_dir = os.path.join(FLAGS.out_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)

        # create model
        model = models.MR_UAE()

        # load dataset
        if FLAGS.data_source == 'knees':
            train_ds = dataloader.read_knees(FLAGS.data_dir, 'small_train', FLAGS.data_shape, num_compressed_coils=FLAGS.num_compressed_coils, emulate_single_coil=FLAGS.emulate_single_coil, quadratic_phase_factor=FLAGS.quadratic_phase_factor, return_img=not FLAGS.simulate_from_kspace, return_kspace=FLAGS.simulate_from_kspace)
            val_ds = dataloader.read_knees(FLAGS.data_dir, 'small_validate', FLAGS.data_shape, num_compressed_coils=FLAGS.num_compressed_coils, emulate_single_coil=FLAGS.emulate_single_coil, quadratic_phase_factor=FLAGS.quadratic_phase_factor, return_img=not FLAGS.simulate_from_kspace, return_kspace=FLAGS.simulate_from_kspace)
        elif FLAGS.data_source == 'knees_full':
            train_ds = dataloader.read_knees(FLAGS.data_dir, 'train', FLAGS.data_shape, num_compressed_coils=FLAGS.num_compressed_coils, emulate_single_coil=FLAGS.emulate_single_coil, quadratic_phase_factor=FLAGS.quadratic_phase_factor, return_img=not FLAGS.simulate_from_kspace, return_kspace=FLAGS.simulate_from_kspace)
            val_ds = dataloader.read_knees(FLAGS.data_dir, 'val', FLAGS.data_shape, num_compressed_coils=FLAGS.num_compressed_coils, emulate_single_coil=FLAGS.emulate_single_coil, quadratic_phase_factor=FLAGS.quadratic_phase_factor, return_img=not FLAGS.simulate_from_kspace, return_kspace=FLAGS.simulate_from_kspace)
        elif FLAGS.data_source == 'fastmri':
            train_ds = dataloader.read_fastmri(FLAGS.data_dir, 'multicoil_train', FLAGS.data_shape)
            val_ds = dataloader.read_fastmri(FLAGS.data_dir, 'multicoil_val', FLAGS.data_shape)
        elif FLAGS.data_source == 'fastmri_small':
            train_ds = dataloader.read_fastmri(FLAGS.data_dir, 'small_multicoil_train', FLAGS.data_shape)
            val_ds = dataloader.read_fastmri(FLAGS.data_dir, 'small_multicoil_val', FLAGS.data_shape)
        elif FLAGS.data_source == 'modl_brain':
            train_ds = dataloader.read_modl_brain(FLAGS.data_dir, 'train')
            val_ds = dataloader.read_modl_brain(FLAGS.data_dir, 'val')
        else:
            raise ValueError('data_source %s not supported' % FLAGS.data_source)

        # batch datasets (and shuffle train dataset)
        if FLAGS.single_batch_training:
            train_ds = train_ds.batch(FLAGS.batch_size).take(1).cache().prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.batch(FLAGS.batch_size).take(1).cache().prefetch(tf.data.AUTOTUNE)
        else:
            train_ds = train_ds.shuffle(10*FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)

        # create summary writers
        train_summary_writer = tf.summary.create_file_writer(summary_dir_train)
        val_summary_writer = tf.summary.create_file_writer(summary_dir_val)

        # start training
        train(model, train_ds, val_ds, train_summary_writer, val_summary_writer, ckpt_dir, best_ckpt_dir, image_dir)
    

def train(model, train_ds, val_ds, train_summary_writer, val_summary_writer, ckpt_dir, best_ckpt_dir, image_dir):

    # check if datasets are empty
    try:
        _ = train_ds.take(1).get_single_element()
    except Exception:
        raise ValueError('Train dataset is empty. Check if the data directory is correct')
    try:
        _ = val_ds.take(1).get_single_element()
    except Exception:
        raise ValueError('Val dataset is empty. Check if the data directory is correct')

    # optimizer
    if FLAGS.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr)
    elif FLAGS.optimizer == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=FLAGS.lr)
    else:
        raise ValueError('Unsupported optimizer.')
    
    # optimizer for encoder
    if FLAGS.lr_enc is not None:
        if FLAGS.optimizer == 'adam':
            optimizer_enc = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr_enc)
        elif FLAGS.optimizer == 'sgd':
            optimizer_enc = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr_enc)
        elif FLAGS.optimizer == 'adamax':
            optimizer_enc = tf.keras.optimizers.Adamax(learning_rate=FLAGS.lr_enc)
        else:
            raise ValueError('Unsupported optimizer.')

    # loss
    if FLAGS.loss == 'mse':
        # mse (l2) for Gaussian observation model
        loss = tf.keras.losses.MeanSquaredError()
    elif FLAGS.loss == 'mae':
        # mae (l1) for Laplacian observation model
        loss = tf.keras.losses.MeanAbsoluteError()
    else:
        raise ValueError('Unsupported loss function.')

    # metrics
    metrics = [
        mrmetrics.MeanAbsoluteComplexError(),
        mrmetrics.MeanSquaredComplexError(),
        mrmetrics.NormalizedRootMeanSquaredComplexError(),
        mrmetrics.PSNR_Complex(),
        mrmetrics.SSIM(),
        #mrmetrics.SSIM_Multiscale()
    ]
    # specify higher or lower is better for the metric
    metric_comparison_functions = [
        tf.less,
        tf.less,
        tf.less,
        tf.greater,
        tf.greater,
        #tf.greater
    ]

    metric_comp = list(zip(metrics, metric_comparison_functions)) # shallow copy is what we need here
    
    # keep record of best metrics during training
    best_metrics = {metric.name: None for metric in metrics}
    # path for saving/restoring best metric values
    best_metrics_path = os.path.join(best_ckpt_dir, 'best_metrics.json')

    # train step function
    @tf.function
    def train_step(inp, trgt):
        with tf.GradientTape() as tape:
            rec = model(inp, training=True)
            loss_value = loss(trgt, rec)
        grads = tape.gradient(loss_value, model.trainable_weights)
        if FLAGS.lr_enc is not None:
            enc_updates = [(grad, var) for grad, var in zip(grads, model.trainable_weights) if 'phi' in var.name]
            dec_updates = [(grad, var) for grad, var in zip(grads, model.trainable_weights) if 'phi' not in var.name]
            optimizer_enc.apply_gradients(enc_updates)
            optimizer.apply_gradients(dec_updates)
        else:
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # tf.print('here')
        # for metric in metrics:
        #     metric(trgt, rec)
        return rec, loss_value
    
    # val step function
    @tf.function
    def val_step(inp):
        rec = model(inp, training=False)
        return rec

    # load latest checkpoint if exists
    # note that the optimizer variables are restored when apply_gradients is called during training
    start_epoch = 0 # update it if checkpoint exists
    ckpt_step = tf.Variable(0) # initialize to zero
    if FLAGS.lr_enc is not None:
        ckpt = tf.train.Checkpoint(step=ckpt_step, optimizer=optimizer, optimizer_enc=optimizer_enc, model=model)
        best_ckpt = tf.train.Checkpoint(step=ckpt_step, optimizer=optimizer, optimizer_enc=optimizer_enc, model=model)        
    else:
        ckpt = tf.train.Checkpoint(step=ckpt_step, optimizer=optimizer, model=model)
        best_ckpt = tf.train.Checkpoint(step=ckpt_step, optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=FLAGS.max_checkpoints)
    latest_ckpt_path = ckpt_manager.latest_checkpoint
    if latest_ckpt_path is not None:
        ckpt.restore(latest_ckpt_path)
        restored_epoch = int(latest_ckpt_path.split('-')[-1]) # epoch number is at the end
        print('Restored checkpoint from epoch %d:\n\t%s'%(restored_epoch, latest_ckpt_path))
        start_epoch = restored_epoch + 1
        print('Calculating validation metrics from the loaded model ...')
        # Run validation loop at the end of each epoch
        for (input_batch, target_batch) in val_ds:
            recon_batch = val_step(input_batch)
            # update val metrics
            for metric in metrics:
                metric(target_batch, recon_batch)
        # Update best metrics
        for metric in metrics:
            best_metrics[metric.name] = metric.result().numpy().item() # convert native python type for json
            metric.reset_states()
        print('Calculated validation metrics from the loaded model ...')
        # if json file for best metrics exists, use that for updating best metrics dict
        if os.path.exists(best_metrics_path):
            with open(best_metrics_path, 'r') as fp:
                best_metrics = json.load(fp)
            print('Loaded best metric values from %s'%best_metrics_path)
        
    # Training Loop
    num_batches = None # unknown number of batches initially
    for epoch in range(start_epoch, FLAGS.num_epochs):
        print('\nEpoch %d/%d' % (epoch, FLAGS.num_epochs))
        progbar = tf.keras.utils.Progbar(num_batches, 
                                        stateful_metrics=[metric.name for metric in metrics])

        # Iterate over the batches in dataset
        for step, (input_batch, target_batch) in enumerate(train_ds):
            recon_batch, loss_value = train_step(input_batch, target_batch)
            progbar.update(step+1, values=[('Training Loss', tf.math.real(loss_value))], finalize=False)

            # Update training metrics
            for metric in metrics:
                metric(target_batch, recon_batch)
            # Log training metrics
            progbar.update(step+1, values=[(metric.name, metric.result()) for metric in metrics], finalize=False)
            # Update step counter in checkpoint (in case step number is needed)
            ckpt.step.assign_add(1) # this also updates best_step.step as they are tied to the same tf.Variable object
        # update total number of batches after first epoch
        num_batches = step + 1

        # Update training summaries at the end of each epoch
        with train_summary_writer.as_default():
            for metric in metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)
                # reset training metrics at the end of each epoch
                metric.reset_states()
            # save the encoded coordinates plot in disk and write it as a summary
            coords_figure = model.plot_coords()
            coords_fpath = os.path.join(image_dir, 'coords_%d.eps' %(epoch))
            coords_figure.savefig(coords_fpath, format='eps')
            coords_fpath = os.path.join(image_dir, 'coords_%d.png' %(epoch))
            coords_figure.savefig(coords_fpath, format='png')
            tf.summary.image('0) Encoded Coordinates', plot_to_image(coords_figure), step=epoch)
            # plt.close(coords_figure)
            # write other image summaries (use the last batch)
            tf.summary.image('1) Ground Truth Magnitude', tf.abs(target_batch[...,tf.newaxis]), step=epoch, max_outputs=3)
            tf.summary.image('2) Reconstruction Magnitude', tf.abs(recon_batch[...,tf.newaxis]), step=epoch, max_outputs=3)
            tf.summary.image('3) Reconstruction Error x 20', 20*(tf.abs(target_batch[...,tf.newaxis] - recon_batch[...,tf.newaxis])), step=epoch, max_outputs=3)
            tf.summary.image('4) Zero Filled Reconstruction Magnitude', tf.expand_dims(tf.abs(model.zerofill_recon(*input_batch)), -1), step=epoch, max_outputs=3)
            tf.summary.image('5) Ground Truth Phase', tf.math.angle(target_batch[...,tf.newaxis]), step=epoch, max_outputs=3)
            tf.summary.image('6) Reconstruction Phase', tf.math.angle(recon_batch[...,tf.newaxis]), step=epoch, max_outputs=3)
            psfs_figure = model.plot_psfs()
            tf.summary.image('7) Point Spread Function', plot_to_image(psfs_figure), step=epoch)
            coords_and_psfs_figure = model.plot_coords_and_psfs()
            tf.summary.image('8) Coordinates and Point Spread Function', plot_to_image(coords_and_psfs_figure), step=epoch)
        
        # Run validation loop every 'eval_every' epochs
        if epoch % FLAGS.eval_every == 0:
            for (input_batch, target_batch) in val_ds:
                recon_batch = val_step(input_batch)
                # recon_batch = model(input_batch, training=False)
                # update val metrics
                for metric in metrics:
                    metric(target_batch, recon_batch)
            # Log validation metrics
            progbar.update(num_batches, values=[(metric.name + '_val', metric.result()) for metric in metrics], finalize=True)
                
            # Update validation summaries
            with val_summary_writer.as_default():
                for metric in metrics:
                    tf.summary.scalar(metric.name, metric.result(), step=epoch)
                # write image summaries (use the last batch)
                tf.summary.image('1) Ground Truth Magnitude', tf.abs(target_batch[...,tf.newaxis]), step=epoch, max_outputs=3)
                tf.summary.image('2) Reconstruction Magnitude', tf.abs(recon_batch[...,tf.newaxis]), step=epoch, max_outputs=3)
                tf.summary.image('3) Reconstruction Error x 20', 20*(tf.abs(target_batch[...,tf.newaxis] - recon_batch[...,tf.newaxis])), step=epoch, max_outputs=3)
                tf.summary.image('4) Zero Filled Reconstruction Magnitude', tf.expand_dims(tf.abs(model.zerofill_recon(*input_batch)), -1), step=epoch, max_outputs=3)
                tf.summary.image('5) Ground Truth Phase', tf.math.angle(target_batch[...,tf.newaxis]), step=epoch, max_outputs=3)
                tf.summary.image('6) Reconstruction Phase', tf.math.angle(recon_batch[...,tf.newaxis]), step=epoch, max_outputs=3)

            # Update best checkpoint if valiadation metric has improved
            for metric, comp_fun in metric_comp:
                if best_metrics[metric.name] is None or comp_fun(metric.result(), best_metrics[metric.name]):
                    best_metrics[metric.name] = metric.result().numpy().item() # convert to native python type for json
                    file_prefix = os.path.join(best_ckpt_dir, 'best_%s'%metric.name)
                    best_ckpt.write(file_prefix=file_prefix) # use write instead of save to avoid using the ckpt counter
                    # update json file that records best metric values so far
                    with open(best_metrics_path, 'w') as fp:
                        json.dump(best_metrics, fp)
                # reset validation metrics after computing the result
                metric.reset_states()
        else:
            # if not running evaluation finalize progbar
            progbar.update(num_batches, finalize=True)


        # Save epoch checkpoint
        if epoch % FLAGS.save_ckpt_every == 0:
            save_path = ckpt_manager.save(epoch)
            print('Saved checkpoint at epoch %d: %s'%(epoch, save_path))



def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  figure.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

if __name__ == '__main__':
    app.run(main)