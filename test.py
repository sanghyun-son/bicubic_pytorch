from os import path
import unittest
from scipy import io

import core
import timer

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

        if cuda.is_available():
            self.test_cuda = True
            self.input_square_cuda = self.input_square.cuda()
            self.input_rect_cuda = self.input_rect.cuda()
            self.input_small_cuda = self.input_small.cuda()
            self.input_topleft_cuda = self.input_topleft.cuda()
            self.input_bottomright_cuda = self.input_bottomright.cuda()
        else:
            self.test_cuda = False

        # You can use different functions for testing your implementation.
        self.imresize = core.imresize
        self.eps = 1e-4

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
        #print(x)
        #print(y)
        diff = torch.norm(x.cpu().float() - y.cpu().float(), 2).item()
        assert diff < self.eps, 'Difference is not negligible!: {}'.format(diff)

    def test_down_down_noaa(self) -> None:
        with timer.Timer('(8, 8) to (3, 4) without AA: {}'):
            x = self.imresize(self.input_square, side=(3, 4))

        self.check_diff(x, 'down_down_noaa')

    def test_cuda_down_down_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with timer.Timer('(8, 8) to (3, 4) without AA using CUDA: {}'):
            x = self.imresize(self.input_square_cuda, side=(3, 4))

        self.check_diff(x, 'down_down_noaa')

    def test_down_down_irregular_noaa(self) -> None:
        with timer.Timer('(8, 8) to (5, 7) without AA: {}'):
            x = self.imresize(self.input_square, side=(5, 7))

        self.check_diff(x, 'down_down_irregular_noaa')

    def test_cuda_down_down_irregular_noaa(self) -> None:
        if self.test_cuda is False:
            return

        with timer.Timer('(8, 8) to (5, 7) without AA using CUDA: {}'):
            x = self.imresize(self.input_square_cuda, side=(5, 7))

        self.check_diff(x, 'down_down_irregular_noaa')

    def test_up_up_topleft(self) -> None:
        with timer.Timer('(4, 4) topleft to (5, 5) without AA: {}'):
            x = self.imresize(self.input_topleft, side=(5, 5))

        self.check_diff(x, 'up_up_topleft_noaa')

    def test_cuda_up_up_topleft(self) -> None:
        if self.test_cuda is False:
            return

        with timer.Timer('(4, 4) topleft to (5, 5) without AA using CUDA: {}'):
            x = self.imresize(self.input_topleft_cuda, side=(5, 5))

        self.check_diff(x, 'up_up_topleft_noaa')

    def test_up_up_bottomright(self) -> None:
        with timer.Timer('(4, 4) bottomright to (5, 5) without AA: {}'):
            x = self.imresize(self.input_bottomright, side=(5, 5))

        self.check_diff(x, 'up_up_bottomright_noaa')

    def test_cuda_up_up_bottomright(self) -> None:
        if self.test_cuda is False:
            return

        with timer.Timer('(4, 4) bottomright to (5, 5) without AA using CUDA: {}'):
            x = self.imresize(self.input_bottomright_cuda, side=(5, 5))

        self.check_diff(x, 'up_up_bottomright_noaa')

    def test_up_up_irregular(self) -> None:
        with timer.Timer('(8, 8) to (11, 13) without AA: {}'):
            x = self.imresize(self.input_square, side=(11, 13))

        self.check_diff(x, 'up_up_irregular_noaa')

    def test_cuda_up_up_irregular(self) -> None:
        if self.test_cuda is False:
            return

        with timer.Timer('(8, 8) to (11, 13) without AA using CUDA: {}'):
            x = self.imresize(self.input_square_cuda, side=(11, 13))

        self.check_diff(x, 'up_up_irregular_noaa')


if __name__ == '__main__':
    unittest.main()
