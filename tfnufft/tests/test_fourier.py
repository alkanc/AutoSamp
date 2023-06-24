
import tensorflow as tf
import numpy as np
import sigpy as sp
from nufft import fourier
# import numpy as np
# import numpy.testing as npt
# from sigpy import fourier, util


class TestFourier(tf.test.TestCase):

    def test_fft(self):
        input = np.array([0, 1, 0], dtype=np.complex64)
        self.assertAllClose(fourier.fft(input, fftdim=1),
                            tf.ones(3) / 3**0.5, atol=1e-5)

        input = np.array([1, 1, 1], dtype=np.complex64)
        self.assertAllClose(fourier.fft(input, fftdim=1),
                            [0, 3**0.5, 0], atol=1e-5)

        input = np.random.normal(size=[4, 5, 6, 2]).astype(np.float32)
        input = input[...,0] + 1j* input[...,1]
        self.assertAllClose(fourier.fft(input, fftdim=3),
                            np.fft.fftshift(np.fft.fftn(
                                np.fft.ifftshift(input), norm='ortho')),
                            atol=1e-5)

        input = np.array([0, 1, 0], dtype=np.complex64)
        self.assertAllClose(fourier.fft(input, fftdim=1, oshape=[5]),
                            np.ones(5) / 5**0.5, atol=1e-5)

    # def test_fft_dtype(self):

    #     for dtype in [np.complex64, np.complex128]:
    #         input = np.array([0, 1, 0], dtype=dtype)
    #         output = fourier.fft(input, fftdim=1)

    #         self.assertDTypeEqual(output, dtype)

    def test_ifft(self):
        input = np.array([0, 1, 0], dtype=np.complex64)
        self.assertAllClose(fourier.ifft(input, fftdim=1),
                            np.ones(3) / 3 ** 0.5, atol=1e-5)

        input = np.array([1, 1, 1], dtype=np.complex64)
        self.assertAllClose(fourier.ifft(input, fftdim=1),
                            [0, 3**0.5, 0], atol=1e-5)

        input = np.random.normal(size=[4, 5, 6, 2]).astype(np.float32)
        input = input[...,0] + 1j* input[...,1]
        self.assertAllClose(fourier.ifft(input, fftdim=3),
                            np.fft.fftshift(np.fft.ifftn(
                                np.fft.ifftshift(input), norm='ortho')),
                            atol=1e-5)

        input = np.array([0, 1, 0], dtype=np.complex64)
        self.assertAllClose(fourier.ifft(input, fftdim=1, oshape=[5]),
                            np.ones(5) / 5**0.5, atol=1e-5)

    def test_nufft(self):

        # Check deltas
        input = np.array([0, 1, 0], np.complex64)  # delta
        coord = np.array([[-1], [0], [1]], np.float32)

        self.assertAllClose(fourier.nufft(input, coord),
                            np.array([1.0, 1.0, 1.0]) / (3**0.5),
                            atol=0.01, rtol=0.01)

        input = np.array([0, 0, 1, 0], np.complex64)  # delta
        coord = np.array([[-2], [-1], [0], [1]], np.float32)

        self.assertAllClose(fourier.nufft(input, coord),
                            np.array([1.0, 1.0, 1.0, 1.0]) / (4**0.5),
                            atol=0.01, rtol=0.01)

        input = np.array([0, 0, 1, 0, 0], np.complex64)  # delta
        coord = np.array([[-2], [-1], [0], [1], [2]], np.float32)

        self.assertAllClose(fourier.nufft(input, coord),
                            np.ones(5) / (5**0.5),
                            atol=0.01, rtol=0.01)

        # Check shifted delta
        input = np.array([0, 0, 1], np.complex64)  # shifted delta
        coord = np.array([[-1], [0], [1]], np.float32)

        w = np.exp(-1j * 2.0 * np.pi / 3.0)
        self.assertAllClose(fourier.nufft(input, coord),
                            np.array([w.conjugate(), 1.0, w]) / (3**0.5),
                            atol=0.01, rtol=0.01)

        input = np.array([0, 0, 0, 1], np.complex64)  # delta
        coord = np.array([[-2], [-1], [0], [1]], np.float32)

        w = np.exp(-1j * 2.0 * np.pi / 4.0)
        self.assertAllClose(
            fourier.nufft(input, coord),
            np.array(
                [w.conjugate()**2, w.conjugate(), 1.0, w]) / (4**0.5),
            atol=0.01, rtol=0.01)

    def test_nufft_nd(self):

        input = np.array([[0], [1], [0]], np.complex64)
        coord = np.array([[-1, 0],
                            [0, 0],
                            [1, 0]], np.float32)

        self.assertAllClose(fourier.nufft(input, coord),
                            np.array([1.0, 1.0, 1.0]) / 3**0.5,
                            atol=0.01, rtol=0.01)

    def test_nufft_adjoint(self):

        # Check deltas
        oshape = [3]
        input = np.array([1.0, 1.0, 1.0], dtype=np.complex64) / 3**0.5
        coord = np.array([[-1], [0], [1]], np.float32)

        self.assertAllClose(fourier.nufft_adjoint(input, coord, oshape),
                            np.array([0, 1, 0]),
                            atol=0.01, rtol=0.01)

        oshape = [4]
        input = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex64) / 4**0.5
        coord = np.array([[-2], [-1], [0], [1]], np.float32)

        self.assertAllClose(fourier.nufft_adjoint(input, coord, oshape),
                            np.array([0, 0, 1, 0], np.complex64),
                            atol=0.01, rtol=0.01)

        oshape = [5]
        input = np.ones(5, dtype=np.complex64) / 5**0.5
        coord = np.array([[-2], [-1], [0], [1], [2]], np.float32)

        self.assertAllClose(fourier.nufft_adjoint(input, coord, oshape),
                            np.array([0, 0, 1, 0, 0], np.complex64),
                            atol=0.01, rtol=0.01)

        # Check shifted delta
        oshape = [3]
        w = np.exp(-1j * 2.0 * np.pi / 3.0)
        input = np.array([w.conjugate(), 1.0, w], np.complex64) / 3**0.5
        coord = np.array([[-1], [0], [1]], np.float32)

        self.assertAllClose(fourier.nufft_adjoint(input, coord, oshape),
                            np.array([0, 0, 1], np.complex64),
                            atol=0.01, rtol=0.01)

        oshape = [4]
        w = np.exp(-1j * 2.0 * np.pi / 4.0)
        input = np.array([w.conjugate()**2, w.conjugate(), 1.0, w], np.complex64) / 4**0.5
        coord = np.array([[-2], [-1], [0], [1]], np.float32)

        self.assertAllClose(fourier.nufft_adjoint(input, coord, oshape),
                            np.array([0, 0, 0, 1], np.complex64),
                            atol=0.01, rtol=0.01)

    def test_nufft_adjoint_nd(self):
        
        oshape = [3, 1]

        input = np.array([1.0, 1.0, 1.0], dtype=np.complex64) / 3**0.5
        coord = np.array([[-1, 0],
                            [0, 0],
                            [1, 0]], np.float32)

        # res = fourier.nufft_adjoint(input, coord, oshape)
        # print(res)
        # print(res.device)

        self.assertAllClose(fourier.nufft_adjoint(input, coord, oshape),
                            np.array([[0], [1], [0]], np.complex64),
                            atol=0.01, rtol=0.01)

    def test_nufft_normal(self):

        # Check delta
        oshape = [3]
        input = np.array([0.0, 1.0, 0.0], dtype=np.complex64)
        coord = np.array([[-1], [0], [1]], np.float32)

        self.assertAllClose(
            fourier.nufft_adjoint(fourier.nufft(
                input, coord), coord, oshape), np.array([0, 1, 0]),
            atol=0.01, rtol=0.01)

        # Check delta scale
        oshape = [3]
        input = np.array([0.0, 1.0, 0.0], dtype=np.complex64)
        coord = np.array([[-1], [-0.5], [0], [0.5], [1]], np.float32)

        self.assertAllClose(
            fourier.nufft_adjoint(
                fourier.nufft(
                    input,
                    coord),
                coord,
                oshape)[
                len(input) // 2],
            5 / 3,
            atol=0.01,
            rtol=0.01)

    def test_nufft_ndft(self):

        n = 5
        w = np.exp(-1j * 2 * np.pi / n)
        coord = np.array([[-2], [0], [0.1]], dtype=np.float32)
        w2 = w**-2
        w1 = w**0.1
        A = np.array([[w2**-2, w2**-1, 1, w2, w2**2],
                      [1, 1, 1, 1, 1],
                      [w1**-2, w1**-1, 1, w1, w1**2]], dtype=np.complex64) / n**0.5

        input = np.eye(n, dtype=np.complex64)
        self.assertAllClose(A.transpose(), fourier.nufft(
            input, coord), atol=0.01, rtol=0.01)

        n = 6
        w = np.exp(-1j * 2 * np.pi / n)
        coord = np.array([[-2], [0], [0.1]], dtype=np.float32)
        w2 = w**-2
        w1 = w**0.1
        A = np.array([[w2**-3, w2**-2, w2**-1, 1, w2, w2**2],
                      [1, 1, 1, 1, 1, 1],
                      [w1**-3, w1**-2, w1**-1, 1, w1, w1**2]]) / n**0.5

        input = np.eye(n, dtype=np.complex64)
        self.assertAllClose(A.transpose(), fourier.nufft(
            input, coord), atol=0.1, rtol=0.1)

    ############################
    ## precomputed kernel tests
    ############################

    def test_nufft_precomputed(self):

        # Check deltas
        input = np.array([0, 1, 0], np.complex64)  # delta
        coord = np.array([[-1], [0], [1]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)

        self.assertAllClose(nufft(input),
                            np.array([1.0, 1.0, 1.0]) / (3**0.5),
                            atol=0.01, rtol=0.01)

        input = np.array([0, 0, 1, 0], np.complex64)  # delta
        coord = np.array([[-2], [-1], [0], [1]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)

        self.assertAllClose(nufft(input),
                            np.array([1.0, 1.0, 1.0, 1.0]) / (4**0.5),
                            atol=0.01, rtol=0.01)

        input = np.array([0, 0, 1, 0, 0], np.complex64)  # delta
        coord = np.array([[-2], [-1], [0], [1], [2]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)

        self.assertAllClose(nufft(input),
                            np.ones(5) / (5**0.5),
                            atol=0.01, rtol=0.01)

        # Check shifted delta
        input = np.array([0, 0, 1], np.complex64)  # shifted delta
        coord = np.array([[-1], [0], [1]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)

        w = np.exp(-1j * 2.0 * np.pi / 3.0)
        self.assertAllClose(nufft(input),
                            np.array([w.conjugate(), 1.0, w]) / (3**0.5),
                            atol=0.01, rtol=0.01)

        input = np.array([0, 0, 0, 1], np.complex64)  # delta
        coord = np.array([[-2], [-1], [0], [1]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)

        w = np.exp(-1j * 2.0 * np.pi / 4.0)
        self.assertAllClose(
            nufft(input),
            np.array(
                [w.conjugate()**2, w.conjugate(), 1.0, w]) / (4**0.5),
            atol=0.01, rtol=0.01)


    def test_nufft_nd_precomputed(self):

        input = np.array([[0], [1], [0]], np.complex64)
        coord = np.array([[-1, 0],
                            [0, 0],
                            [1, 0]], np.float32)
        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)

        self.assertAllClose(nufft(input),
                            np.array([1.0, 1.0, 1.0]) / 3**0.5,
                            atol=0.01, rtol=0.01)

    def test_nufft_adjoint_precomputed(self):

        # Check deltas
        oshape = [3]
        input = np.array([1.0, 1.0, 1.0], dtype=np.complex64) / 3**0.5
        coord = np.array([[-1], [0], [1]], np.float32)
        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(nufft_adjoint(input),
                            np.array([0, 1, 0]),
                            atol=0.01, rtol=0.01)

        oshape = [4]
        input = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex64) / 4**0.5
        coord = np.array([[-2], [-1], [0], [1]], np.float32)
        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(nufft_adjoint(input),
                            np.array([0, 0, 1, 0], np.complex64),
                            atol=0.01, rtol=0.01)

        oshape = [5]
        input = np.ones(5, dtype=np.complex64) / 5**0.5
        coord = np.array([[-2], [-1], [0], [1], [2]], np.float32)
        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(nufft_adjoint(input),
                            np.array([0, 0, 1, 0, 0], np.complex64),
                            atol=0.01, rtol=0.01)

        # Check shifted delta
        oshape = [3]
        w = np.exp(-1j * 2.0 * np.pi / 3.0)
        input = np.array([w.conjugate(), 1.0, w], np.complex64) / 3**0.5
        coord = np.array([[-1], [0], [1]], np.float32)
        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(nufft_adjoint(input),
                            np.array([0, 0, 1], np.complex64),
                            atol=0.01, rtol=0.01)

        oshape = [4]
        w = np.exp(-1j * 2.0 * np.pi / 4.0)
        input = np.array([w.conjugate()**2, w.conjugate(), 1.0, w], np.complex64) / 4**0.5
        coord = np.array([[-2], [-1], [0], [1]], np.float32)
        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(nufft_adjoint(input),
                            np.array([0, 0, 0, 1], np.complex64),
                            atol=0.01, rtol=0.01)


    def test_nufft_adjoint_nd_precomputed(self):
        
        oshape = [3, 1]

        input = np.array([1.0, 1.0, 1.0], dtype=np.complex64) / 3**0.5
        coord = np.array([[-1, 0],
                            [0, 0],
                            [1, 0]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(nufft_adjoint(input),
                            np.array([[0], [1], [0]], np.complex64),
                            atol=0.01, rtol=0.01)

    def test_nufft_normal_precomputed(self):

        # Check delta
        oshape = [3]
        input = np.array([0.0, 1.0, 0.0], dtype=np.complex64)
        coord = np.array([[-1], [0], [1]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft = fourier.nufft_from_obj(nufft_obj)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(
            nufft_adjoint(nufft(input)), np.array([0, 1, 0]),
            atol=0.01, rtol=0.01)

        # Check delta scale
        oshape = [3]
        input = np.array([0.0, 1.0, 0.0], dtype=np.complex64)
        coord = np.array([[-1], [-0.5], [0], [0.5], [1]], np.float32)

        nufft_obj = fourier.get_nufft_obj(coord, oshape)
        nufft = fourier.nufft_from_obj(nufft_obj)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(
            nufft_adjoint(nufft(input))[
                len(input) // 2],
            5 / 3,
            atol=0.01,
            rtol=0.01)

    def test_nufft_ndft_precomputed(self):

        n = 5
        w = np.exp(-1j * 2 * np.pi / n)
        coord = np.array([[-2], [0], [0.1]], dtype=np.float32)
        w2 = w**-2
        w1 = w**0.1
        A = np.array([[w2**-2, w2**-1, 1, w2, w2**2],
                      [1, 1, 1, 1, 1],
                      [w1**-2, w1**-1, 1, w1, w1**2]], dtype=np.complex64) / n**0.5

        input = np.eye(n, dtype=np.complex64)

        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)

        self.assertAllClose(A.transpose(), nufft(
            input), atol=0.01, rtol=0.01)

        n = 6
        w = np.exp(-1j * 2 * np.pi / n)
        coord = np.array([[-2], [0], [0.1]], dtype=np.float32)
        w2 = w**-2
        w1 = w**0.1
        A = np.array([[w2**-3, w2**-2, w2**-1, 1, w2, w2**2],
                      [1, 1, 1, 1, 1, 1],
                      [w1**-3, w1**-2, w1**-1, 1, w1, w1**2]]) / n**0.5

        input = np.eye(n, dtype=np.complex64)

        nufft_obj = fourier.get_nufft_obj(coord, input.shape)
        nufft = fourier.nufft_from_obj(nufft_obj)
        nufft_adjoint = fourier.nufft_adjoint_from_obj(nufft_obj)


        self.assertAllClose(A.transpose(), nufft(
            input), atol=0.1, rtol=0.1)


if __name__ == '__main__':
    tf.test.main()