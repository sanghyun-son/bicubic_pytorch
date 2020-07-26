from os import path
import unittest

import core_warp
import utils

import torch
from torch import cuda


class TestWarpBenchmark(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n = 1000
        self.butterfly = utils.get_img(path.join('example', 'butterfly.png'))
        # Batching
        self.butterfly = self.butterfly.repeat(16, 1, 1, 1)
        self.m = torch.Tensor([
            [3.2, 0.016, -68],
            [1.23, 1.7, -54],
            [0.008, 0.0001, 1],
        ])
        if cuda.is_available():
            self.butterfly = self.butterfly.cuda()
            self.m = self.m.cuda()

            with utils.Timer('Warm-up: {}'):
                for _ in range(100):
                    _ = core_warp.warp(
                        self.butterfly,
                        self.m,
                        sizes='auto',
                        kernel='bicubic',
                        fill_value=0,
                    )

                cuda.synchronize()

    def test_warp_nearest(self) -> torch.Tensor:
        with utils.Timer('Nearest warping: {}'):
            for _ in range(self.n):
                _ = core_warp.warp(
                    self.butterfly,
                    self.m,
                    sizes='auto',
                    kernel='nearest',
                    fill_value=0,
                )

            cuda.synchronize()

    def test_warp_bilinear(self) -> torch.Tensor:
        with utils.Timer('Bilinear warping: {}'):
            for _ in range(self.n):
                _ = core_warp.warp(
                    self.butterfly,
                    self.m,
                    sizes='auto',
                    kernel='bilinear',
                    fill_value=0,
                )

            cuda.synchronize()

    def test_warp_bicubic(self) -> torch.Tensor:
        with utils.Timer('Bicubic warping: {}'):
            for _ in range(self.n):
                _ = core_warp.warp(
                    self.butterfly,
                    self.m,
                    sizes='auto',
                    kernel='bicubic',
                    fill_value=0,
                )

            cuda.synchronize()


if __name__ == '__main__':
    unittest.main()