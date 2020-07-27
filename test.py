from os import path
import unittest
from scipy import io

import core
import utils

import torch
from torch import cuda


class TestBicubic(unittest.TestCase):
    '''
    Why do we have to split CUDA?
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_small = torch.arange(16).view(1, 1, 4, 4).float()
        self.input_square = torch.arange(64).view(1, 1, 8, 8).float()
        self.input_rect = torch.arange(80).view(1, 1, 8, 10).float()
        self.input_15x15 = torch.arange(225).view(1, 1, 15, 15).float()
        
        self.input_topleft = torch.zeros(1, 1, 4, 4).float()
        self.input_topleft[..., 0, 0] = 100
        self.input_topleft[..., 1, 0] = 10
        self.input_topleft[..., 0, 1] = 1
        self.input_topleft[..., 0, 3] = 100

        self.input_bottomright = torch.zeros(1, 1, 4, 4).float()
        self.input_bottomright[..., 3, 3] = 100
        self.input_bottomright[..., 2, 3] = 10
        self.input_bottomright[..., 3, 2] = 1
        self.input_bottomright[..., 3, 0] = 100

        self.butterfly = utils.get_img(path.join('example', 'butterfly.png'))

        if cuda.is_available():
            self.test_cuda = True
            self.input_small_cuda = self.input_small.cuda()
            self.input_square_cuda = self.input_square.cuda()
            self.input_rect_cuda = self.input_rect.cuda()
            self.input_15x15_cuda = self.input_15x15.cuda()
            self.input_topleft_cuda = self.input_topleft.cuda()
            self.input_bottomright_cuda = self.input_bottomright.cuda()
            self.butterfly_cuda = self.butterfly.cuda()
        else:
            self.test_cuda = False

        # You can use different functions for testing your implementation.
        self.imresize = core.imresize
        self.eps = 1e-3

    def get_answer(self, case: str) -> torch.Tensor:
        mat = io.loadmat(path.join('test_answer', case + '.mat'))
        tensor = torch.Tensor(mat[case])
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)

        while tensor.dim() < 4:
            tensor.unsqueeze_(0)

        return tensor

    def _check_diff(self, x: torch.Tensor, y: torch.Tensor) -> None:
        diff = torch.norm(x - y, 2).item()
        if diff > self.eps:
            print('Implementation:')
            print(x)
            print('MATLAB reference:')
            print(y)
            raise ArithmeticError(
                'Difference is not negligible!: {}'.format(diff),
            )
        else:
            print('Allowable difference: {:.4e} < {:.4e}'.format(diff, self.eps))

        return

    def check_diff(self, x: torch.Tensor, answer: str) -> None:
        y = self.get_answer(answer).to(dtype=x.dtype, device=x.device)
        self._check_diff(x, y)
        return

    def test_consistency_down_down_x4_large_noaa(self) -> None:
        x = torch.randn(1, 3, 2048, 2048)
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'without AA (Cubic Conv. vs. Naive): {}'):

            x = self.imresize(x, scale=0.25, antialiasing=False)
            y = self.imresize(x, sizes=(512, 512), antialiasing=False)

        self._check_diff(x, y)
        return

    def test_cuda_consistency_down_down_x4_large_noaa(self) -> None:
        if self.test_cuda is False:
            return

        x = torch.randn(1, 3, 2048, 2048).cuda()
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'without AA (Cubic Conv. vs. Naive) using CUDA: {}'):

            x = self.imresize(x, scale=0.25, antialiasing=False)
            y = self.imresize(x, sizes=(512, 512), antialiasing=False)

        self._check_diff(x, y)
        return

    def test_consistency_down_down_x4_large_aa(self) -> None:
        x = torch.randn(1, 3, 2048, 2048)
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'with AA (Cubic Conv. vs. Naive): {}'):

            x = self.imresize(x, scale=0.25, antialiasing=True)
            y = self.imresize(x, sizes=(512, 512), antialiasing=True)

        self._check_diff(x, y)
        return

    def test_cuda_consistency_down_down_x4_large_aa(self) -> None:
        if self.test_cuda is False:
            return

        x = torch.randn(1, 3, 2048, 2048).cuda()
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'with AA (Cubic Conv. vs. Naive) using CUDA: {}'):

            x = self.imresize(x, scale=0.25, antialiasing=True)
            y = self.imresize(x, sizes=(512, 512), antialiasing=True)

        self._check_diff(x, y)
        return

    def test_down_down_x4_large_noaa(self) -> None:
        x = torch.randn(1, 3, 2048, 2048)
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) without AA (Cubic Conv.): {}'):

            x = self.imresize(x, scale=0.25, antialiasing=False)

        return

    def test_cuda_down_down_x4_large_noaa(self) -> None:
        if self.test_cuda is False:
            return

        x = torch.randn(1, 3, 2048, 2048).cuda()
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'without AA using CUDA (Cubic Conv.): {}'):

            x = self.imresize(x, scale=0.25, antialiasing=False)

        return

    def test_down_down_x4_naive_large_noaa(self) -> None:
        x = torch.randn(1, 3, 2048, 2048)
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'without AA (Naive): {}'):

            x = self.imresize(x, sizes=(512, 512), antialiasing=False)

        return

    def test_cuda_down_down_x4_naive_large_noaa(self) -> None:
        if self.test_cuda is False:
            return

        x = torch.randn(1, 3, 2048, 2048).cuda()
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'without AA using CUDA (Naive): {}'):

            x = self.imresize(x, sizes=(512, 512), antialiasing=False)

        return

    def test_down_down_x4_large_aa(self) -> None:
        x = torch.randn(1, 3, 2048, 2048)
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) with AA (Cubic Conv.): {}'):

            x = self.imresize(x, scale=0.25, antialiasing=True)

        return

    def test_cuda_down_down_x4_large_aa(self) -> None:
        if self.test_cuda is False:
            return

        x = torch.randn(1, 3, 2048, 2048).cuda()
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'with AA using CUDA (Cubic Conv.): {}'):

            x = self.imresize(x, scale=0.25, antialiasing=True)

        return

    def test_down_down_x4_naive_large_aa(self) -> None:
        x = torch.randn(1, 3, 2048, 2048)
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) with AA (Naive): {}'):

            x = self.imresize(x, sizes=(512, 512), antialiasing=True)

        return

    def test_cuda_down_down_x4_naive_large_aa(self) -> None:
        if self.test_cuda is False:
            return

        x = torch.randn(1, 3, 2048, 2048).cuda()
        with utils.Timer(
                '(2048, 2048) RGB to (512, 512) '
                'with AA using CUDA (Naive): {}'):

            x = self.imresize(x, sizes=(512, 512), antialiasing=True)

        return

    def test_down_down_small_noaa(self) -> None:
        with utils.Timer('(4, 4) to (3, 3) without AA: {}'):
            x = self.imresize(
                self.input_small, sizes=(3, 3), antialiasing=False,
            )

        self.check_diff(x, 'down_down_small_noaa')
        return

    def test_cuda_down_down_small_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) to (3, 3) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_small_cuda, sizes=(3, 3), antialiasing=False,
            )

        self.check_diff(x, 'down_down_small_noaa')
        return

    def test_down_down_small_aa(self) -> None:
        with utils.Timer('(4, 4) to (3, 3) with AA: {}'):
            x = self.imresize(
                self.input_small, sizes=(3, 3), antialiasing=True,
            )

        self.check_diff(x, 'down_down_small_aa')
        return

    def test_cuda_down_down_small_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) to (3, 3) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_small_cuda, sizes=(3, 3), antialiasing=True,
            )

        self.check_diff(x, 'down_down_small_aa')
        return

    def test_down_down_noaa(self) -> None:
        with utils.Timer('(8, 8) to (3, 4) without AA: {}'):
            x = self.imresize(
                self.input_square, sizes=(3, 4), antialiasing=False,
            )

        self.check_diff(x, 'down_down_noaa')
        return

    def test_cuda_down_down_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (3, 4) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sizes=(3, 4), antialiasing=False,
            )

        self.check_diff(x, 'down_down_noaa')
        return

    def test_down_down_aa(self) -> None:
        with utils.Timer('(8, 8) to (3, 4) with AA: {}'):
            x = self.imresize(
                self.input_square, sizes=(3, 4), antialiasing=True,
            )

        self.check_diff(x, 'down_down_aa')
        return

    def test_cuda_down_down_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (3, 4) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sizes=(3, 4), antialiasing=True,
            )

        self.check_diff(x, 'down_down_aa')
        return

    def test_down_down_irregular_noaa(self) -> None:
        with utils.Timer('(8, 8) to (5, 7) without AA: {}'):
            x = self.imresize(
                self.input_square, sizes=(5, 7), antialiasing=False,
            )

        self.check_diff(x, 'down_down_irregular_noaa')
        return

    def test_cuda_down_down_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (5, 7) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sizes=(5, 7), antialiasing=False,
            )

        self.check_diff(x, 'down_down_irregular_noaa')
        return

    def test_down_down_x2_aa(self) -> None:
        with utils.Timer('(8, 8) to (4, 4) with AA: {}'):
            x = self.imresize(
                self.input_square, scale=(1 / 2), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x2_aa')
        return

    def test_cuda_down_down_x2_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (4, 4) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, scale=(1 / 2), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x2_aa')
        return

    def test_down_down_x3_aa(self) -> None:
        with utils.Timer('(15, 15) to (5, 5) with AA: {}'):
            x = self.imresize(
                self.input_15x15, scale=(1 / 3), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x3_aa')
        return

    def test_cuda_down_down_x3_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(15, 15) to (5, 5) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_15x15_cuda, scale=(1 / 3), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x3_aa')
        return

    def test_down_down_x4_aa(self) -> None:
        with utils.Timer('(8, 8) to (2, 2) with AA: {}'):
            x = self.imresize(
                self.input_square, scale=(1 / 4), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x4_aa')
        return

    def test_cuda_down_down_x4_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (2, 2) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, scale=(1 / 4), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x4_aa')
        return

    def test_down_down_x5_aa(self) -> None:
        with utils.Timer('(15, 15) to (3, 3) with AA: {}'):
            x = self.imresize(
                self.input_15x15, scale=(1 / 5), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x5_aa')
        return

    def test_cuda_down_down_x5_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(15, 15) to (3, 3) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_15x15_cuda, scale=(1 / 5), antialiasing=True,
            )

        self.check_diff(x, 'down_down_x5_aa')
        return

    def test_up_up_topleft_noaa(self) -> None:
        with utils.Timer('(4, 4) topleft to (5, 5) without AA: {}'):
            x = self.imresize(
                self.input_topleft, sizes=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_topleft_noaa')
        return

    def test_cuda_up_up_topleft_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) topleft to (5, 5) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_topleft_cuda, sizes=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_topleft_noaa')
        return

    def test_up_up_bottomright_noaa(self) -> None:
        with utils.Timer('(4, 4) bottomright to (5, 5) without AA: {}'):
            x = self.imresize(
                self.input_bottomright, sizes=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_bottomright_noaa')
        return

    def test_cuda_up_up_bottomright_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) bottomright to (5, 5) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_bottomright_cuda, sizes=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_bottomright_noaa')
        return

    def test_up_up_irregular_noaa(self) -> None:
        with utils.Timer('(8, 8) to (11, 13) without AA: {}'):
            x = self.imresize(
                self.input_square, sizes=(11, 13), antialiasing=False,
            )

        self.check_diff(x, 'up_up_irregular_noaa')
        return

    def test_cuda_up_up_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (11, 13) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sizes=(11, 13), antialiasing=False,
            )

        self.check_diff(x, 'up_up_irregular_noaa')
        return

    def test_up_up_irregular_aa(self) -> None:
        with utils.Timer('(8, 8) to (11, 13) with AA: {}'):
            x = self.imresize(
                self.input_square, sizes=(11, 13), antialiasing=True,
            )

        self.check_diff(x, 'up_up_irregular_aa')
        return

    def test_cuda_up_up_irregular_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (11, 13) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sizes=(11, 13), antialiasing=True,
            )

        self.check_diff(x, 'up_up_irregular_aa')
        return

    def test_down_down_butterfly_irregular_noaa(self) -> None:
        with utils.Timer('(256, 256) butterfly to (123, 234) without AA: {}'):
            x = self.imresize(
                self.butterfly, sizes=(123, 234), antialiasing=False,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_noaa')
        return

    def test_cuda_down_down_butterfly_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(256, 256) butterfly to (123, 234) without AA using CUDA: {}'):
            x = self.imresize(
                self.butterfly_cuda, sizes=(123, 234), antialiasing=False,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_noaa')
        return

    def test_double_down_down_butterfly_irregular_noaa(self) -> None:
        double = self.butterfly.double()
        with utils.Timer('(256, 256) butterfly (double) to (123, 234) without AA: {}'):
            x = self.imresize(double, sizes=(123, 234), antialiasing=False)

        self.check_diff(x, 'down_down_butterfly_irregular_noaa')
        return

    def test_double_cuda_down_down_butterfly_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        double = self.butterfly_cuda.double()
        with utils.Timer('(256, 256) butterfly (double) to (123, 234) without AA using CUDA: {}'):
            x = self.imresize(double, sizes=(123, 234), antialiasing=False)

        self.check_diff(x, 'down_down_butterfly_irregular_noaa')
        return

    def test_down_down_butterfly_irregular_aa(self) -> None:
        with utils.Timer('(256, 256) butterfly to (123, 234) with AA: {}'):
            x = self.imresize(
                self.butterfly, sizes=(123, 234), antialiasing=True,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_aa')
        return

    def test_cuda_down_down_butterfly_irregular_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(256, 256) butterfly to (123, 234) with AA using CUDA: {}'):
            x = self.imresize(
                self.butterfly_cuda, sizes=(123, 234), antialiasing=True,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_aa')
        return

    def test_up_up_butterfly_irregular_noaa(self) -> None:
        with utils.Timer('(256, 256) butterfly to (1234, 789) without AA: {}'):
            x = self.imresize(
                self.butterfly, sizes=(1234, 789), antialiasing=False,
            )

        self.check_diff(x, 'up_up_butterfly_irregular_noaa')
        return

    def test_cuda_up_up_butterfly_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(256, 256) butterfly to (1234, 789) without AA using CUDA: {}'):
            x = self.imresize(
                self.butterfly_cuda, sizes=(1234, 789), antialiasing=False,
            )

        self.check_diff(x, 'up_up_butterfly_irregular_noaa')
        return


if __name__ == '__main__':
    unittest.main()
