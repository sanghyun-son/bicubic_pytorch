from os import path
import unittest
from scipy import io

import core
import utils

import torch
from torch import cuda


class TestBicubic(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_small = torch.arange(16).view(1, 1, 4, 4).float()
        self.input_square = torch.arange(64).view(1, 1, 8, 8).float()
        self.input_rect = torch.arange(80).view(1, 1, 8, 10).float()
        
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
            self.input_square_cuda = self.input_square.cuda()
            self.input_rect_cuda = self.input_rect.cuda()
            self.input_small_cuda = self.input_small.cuda()
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

    def check_diff(self, x: torch.Tensor, answer: str) -> float:
        y = self.get_answer(answer)
        diff = torch.norm(x.cpu().float() - y.cpu().float(), 2).item()
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

    def test_down_down_small_noaa(self) -> None:
        with utils.Timer('(4, 4) to (3, 3) without AA: {}'):
            x = self.imresize(
                self.input_small, sides=(3, 3), antialiasing=False,
            )

        self.check_diff(x, 'down_down_small_noaa')

    def test_cuda_down_down_small_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) to (3, 3) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_small_cuda, sides=(3, 3), antialiasing=False,
            )

        self.check_diff(x, 'down_down_small_noaa')

    def test_down_down_small_aa(self) -> None:
        with utils.Timer('(4, 4) to (3, 3) with AA: {}'):
            x = self.imresize(
                self.input_small, sides=(3, 3), antialiasing=True,
            )

        self.check_diff(x, 'down_down_small_aa')

    def test_cuda_down_down_small_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) to (3, 3) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_small_cuda, sides=(3, 3), antialiasing=True,
            )

        self.check_diff(x, 'down_down_small_aa')

    def test_down_down_noaa(self) -> None:
        with utils.Timer('(8, 8) to (3, 4) without AA: {}'):
            x = self.imresize(
                self.input_square, sides=(3, 4), antialiasing=False,
            )

        self.check_diff(x, 'down_down_noaa')

    def test_cuda_down_down_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (3, 4) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sides=(3, 4), antialiasing=False,
            )

        self.check_diff(x, 'down_down_noaa')

    def test_down_down_aa(self) -> None:
        with utils.Timer('(8, 8) to (3, 4) with AA: {}'):
            x = self.imresize(
                self.input_square, sides=(3, 4), antialiasing=True,
            )

        self.check_diff(x, 'down_down_aa')

    def test_cuda_down_down_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (3, 4) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sides=(3, 4), antialiasing=True,
            )

        self.check_diff(x, 'down_down_aa')

    def test_down_down_irregular_noaa(self) -> None:
        with utils.Timer('(8, 8) to (5, 7) without AA: {}'):
            x = self.imresize(
                self.input_square, sides=(5, 7), antialiasing=False,
            )

        self.check_diff(x, 'down_down_irregular_noaa')

    def test_cuda_down_down_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (5, 7) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sides=(5, 7), antialiasing=False,
            )

        self.check_diff(x, 'down_down_irregular_noaa')

    def test_up_up_topleft_noaa(self) -> None:
        with utils.Timer('(4, 4) topleft to (5, 5) without AA: {}'):
            x = self.imresize(
                self.input_topleft, sides=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_topleft_noaa')

    def test_cuda_up_up_topleft_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) topleft to (5, 5) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_topleft_cuda, sides=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_topleft_noaa')

    def test_up_up_bottomright_noaa(self) -> None:
        with utils.Timer('(4, 4) bottomright to (5, 5) without AA: {}'):
            x = self.imresize(
                self.input_bottomright, sides=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_bottomright_noaa')

    def test_cuda_up_up_bottomright_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(4, 4) bottomright to (5, 5) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_bottomright_cuda, sides=(5, 5), antialiasing=False,
            )

        self.check_diff(x, 'up_up_bottomright_noaa')

    def test_up_up_irregular_noaa(self) -> None:
        with utils.Timer('(8, 8) to (11, 13) without AA: {}'):
            x = self.imresize(
                self.input_square, sides=(11, 13), antialiasing=False,
            )

        self.check_diff(x, 'up_up_irregular_noaa')

    def test_cuda_up_up_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (11, 13) without AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sides=(11, 13), antialiasing=False,
            )

        self.check_diff(x, 'up_up_irregular_noaa')

    def test_up_up_irregular_aa(self) -> None:
        with utils.Timer('(8, 8) to (11, 13) with AA: {}'):
            x = self.imresize(
                self.input_square, sides=(11, 13), antialiasing=True,
            )

        self.check_diff(x, 'up_up_irregular_aa')

    def test_cuda_up_up_irregular_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(8, 8) to (11, 13) with AA using CUDA: {}'):
            x = self.imresize(
                self.input_square_cuda, sides=(11, 13), antialiasing=True,
            )

        self.check_diff(x, 'up_up_irregular_aa')

    def test_down_down_butterfly_irregular_noaa(self) -> None:
        with utils.Timer('(256, 256) butterfly to (123, 234) without AA: {}'):
            x = self.imresize(
                self.butterfly, sides=(123, 234), antialiasing=False,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_noaa')

    def test_cuda_down_down_butterfly_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(256, 256) butterfly to (123, 234) without AA using CUDA: {}'):
            x = self.imresize(
                self.butterfly_cuda, sides=(123, 234), antialiasing=False,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_noaa')

    def test_down_down_butterfly_irregular_aa(self) -> None:
        with utils.Timer('(256, 256) butterfly to (123, 234) with AA: {}'):
            x = self.imresize(
                self.butterfly, sides=(123, 234), antialiasing=True,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_aa')

    def test_cuda_down_down_butterfly_irregular_aa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(256, 256) butterfly to (123, 234) with AA using CUDA: {}'):
            x = self.imresize(
                self.butterfly_cuda, sides=(123, 234), antialiasing=True,
            )

        self.check_diff(x, 'down_down_butterfly_irregular_aa')

    def test_up_up_butterfly_irregular_noaa(self) -> None:
        with utils.Timer('(256, 256) butterfly to (1234, 789) without AA: {}'):
            x = self.imresize(
                self.butterfly, sides=(1234, 789), antialiasing=False,
            )

        self.check_diff(x, 'up_up_butterfly_irregular_noaa')

    def test_cuda_up_up_butterfly_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with utils.Timer('(256, 256) butterfly to (1234, 789) without AA using CUDA: {}'):
            x = self.imresize(
                self.butterfly_cuda, sides=(1234, 789), antialiasing=False,
            )

        self.check_diff(x, 'up_up_butterfly_irregular_noaa')


if __name__ == '__main__':
    unittest.main()
