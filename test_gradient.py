from os import path
import unittest

import core
import utils

import torch
from torch import nn
from torch.nn import functional as F
from torch import cuda
from torch import optim


class TestGradient(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_iters = 200
        self.lr = 1e-2
        self.input_size = (123, 234)

        if cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.target = utils.get_img(path.join('example', 'butterfly.png'))
        self.target = self.target.to(self.device)
        self.target_size = (self.target.size(-2), self.target.size(-1))

    def test_backpropagation(self) -> None:
        noise = torch.rand(
            1,
            self.target.size(1),
            self.input_size[0],
            self.input_size[1],
            device=self.device,
        )
        noise_p = nn.Parameter(noise, requires_grad=True)
        utils.save_img(noise_p, path.join('example', 'noise_input.png'))
        optimizer = optim.Adam([noise_p], lr=self.lr)

        for i in range(self.n_iters):
            optimizer.zero_grad()
            noise_up = core.imresize(noise_p, size=self.target_size)
            loss = F.mse_loss(noise_up, self.target)
            loss.backward()
            if i == 0 or (i + 1) % 20 == 0:
                print('Iter {:0>4}\tLoss: {:.8f}'.format(i + 1, loss.item()))

            optimizer.step()

        utils.save_img(noise_p, path.join('example', 'noise_optimized.png'))
        assert loss.item() < 1e-2, 'Failed to optimize!'


if __name__ == '__main__':
    unittest.main()