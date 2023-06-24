import time

import numpy as np

import skimage.data

import tensorflow as tf

from tfnufft import fourier

def profile_tfnufft(
        image,
        ktraj,
        im_size,
        device,
        oversamp,
        width,
        use_graph_mode
    ):
    if device == 'CPU':
        num_nuffts = 100
    else:
        num_nuffts = 100
    print(f'Using {device}, graph_mode: {use_graph_mode}, oversamp: {oversamp}, width: {width}')
    device_name = f'/{device}:0'
    with tf.device(device_name):
        image = tf.constant(image)
        if device == 'GPU':
            image = tf.cast(image, tf.complex64)
        ktraj = tf.constant(ktraj)

        get_nufft_obj_op = fourier.get_nufft_obj
        if use_graph_mode:
            get_nufft_obj_op = tf.function(get_nufft_obj_op)

        # warm-up computation
        for _ in range(2):
            nufft_ob = get_nufft_obj_op(ktraj, oshape=im_size, oversamp=oversamp, width=width)

        runtimes = []
        for _ in range(num_nuffts):
            start_time = time.perf_counter()
            nufft_ob = get_nufft_obj_op(ktraj, oshape=im_size, oversamp=oversamp, width=width)
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
        print('precomputation time: {}, std: {}'.format(np.mean(runtimes), np.std(runtimes)))

        # forward_op = lambda img, coord: fourier.nufft(img, coord, oversamp=oversamp, width=width)
        forward_op = fourier.nufft_from_obj(nufft_ob)
        adjoint_op = fourier.nufft_adjoint_from_obj(nufft_ob)
        if use_graph_mode:
            forward_op = tf.function(forward_op)
            adjoint_op = tf.function(adjoint_op)

        # warm-up computation
        for _ in range(2):
            y = forward_op(image)

        runtimes = []
        for _ in range(num_nuffts):
            start_time = time.perf_counter()
            y = forward_op(image)
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
        print('forward average time: {}, std: {}'.format(np.mean(runtimes), np.std(runtimes)))

        # warm-up computation
        for _ in range(2):
            x = adjoint_op(y)

        # run the adjoint speed tests
        runtimes = []
        for _ in range(num_nuffts):
            start_time = time.perf_counter()
            x = adjoint_op(y)
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
        # avg_time = (end_time-start_time) / num_nuffts
        print('backward average time: {}, std: {}'.format(np.mean(runtimes), np.std(runtimes)))



def run_all_profiles():
    print('running profiler...')
    spokelength = 512
    nspokes = 405

    print('problem size (radial trajectory, 2-factor oversampling):')
    print('number of spokes: {}'.format(nspokes))
    print('spokelength: {}'.format(spokelength))

    # create an example to run on
    image = skimage.data.brain()[:1]
    image = image.astype(np.complex64)
    im_size = image.shape[-2:]

    # image = image[None, None, ...]

    # create k-space trajectory
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=1).astype(np.float32)

    # ktraj = ktraj[None, ...]

    profile_tfnufft(image, ktraj, im_size, device='CPU', oversamp=1.25, width=4., use_graph_mode=False)
    profile_tfnufft(image, ktraj, im_size, device='CPU', oversamp=1.25, width=4., use_graph_mode=True)
    profile_tfnufft(image, ktraj, im_size, device='GPU', oversamp=1.25, width=4., use_graph_mode=False)
    profile_tfnufft(image, ktraj, im_size, device='GPU', oversamp=1.25, width=4., use_graph_mode=True)


if __name__ == '__main__':
    run_all_profiles()